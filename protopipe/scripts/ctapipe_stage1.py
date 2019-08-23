""" 
User Tools to process  {R0,R1,DL0}/Event data into DL1/Event data
"""
from collections import OrderedDict, defaultdict, namedtuple
from functools import partial
from pathlib import Path

import numpy as np
import traitlets

import ctapipe
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.core import Component, Container, Field, Tool, ToolConfigurationError
from ctapipe.core.traits import Bool, CaselessStrEnum, Dict, Float, Int, List, Unicode
from ctapipe.image import HillasParameterizationError, hillas_parameters, tailcuts_clean
from ctapipe.image.concentration import concentration
from ctapipe.image.leakage import leakage
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.instrument import CameraGeometry, TelescopeDescription
from ctapipe.io import EventSource, HDF5TableWriter
from ctapipe.io.containers import (
    ConcentrationContainer,
    DL1CameraContainer,
    EventIndexContainer,
    HillasParametersContainer,
    ImageParametersContainer,
    LeakageContainer,
    TelEventIndexContainer,
    TimingParametersContainer,
)
from ctapipe.utils import CutFlow
import tables.filters

RangeTuple = namedtuple("RangeTuple", "min,max")


class Range(traitlets.TraitType):
    """ A Traitlet that accepts a range like (min,max) or [min,max] """

    # TODO: make unit tests.
    default_value = (-np.inf, np.inf)
    info_text = "a tuple or list of (min, max) values, with min<=max"

    def validate(self, obj, value):
        if isinstance(value, tuple) or isinstance(value, list):
            if len(value) == 2:
                if value[0] <= value[1]:
                    return RangeTuple(*value)

        self.error(obj, value)


class DataChecker(Component):

    selection_functions = Dict(
        help="dict of cut name : lambda function in string format to accept (select) data"
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        """
        Manages a set of selection cuts on the same input quantity or class.

        Allows one to access the current set of cuts as a Container  or a boolean value
        (so the results can be output per event if necessary)
        
        Parameters
        ----------
        name : str
            name of this set of cuts
        selector_functions: OrderedDict[str, str]
            dictionary of "cut name" to acceptance function string. 
            The acceptance functions are string representation of a lambda
            function and return True if the input value is *accepted*
        """
        super().__init__(config=config, parent=parent, **kwargs)

        self._selectors = OrderedDict(
            (name, eval(func_str))
            for name, func_str in self.selection_functions.items()
        )

        self._counts_total = 0
        self._counts = np.zeros(len(self._selectors), dtype=np.int)
        self._counts_weighted = np.zeros(len(self._selectors), dtype=np.int)

        # generate a Container that we can use for output somehow...
        self._container = Container()

    def __len__(self):
        return self._counts_total

    @property
    def criteria_names(self):
        return list(self._selectors.keys())

    @property
    def selection_function_strings(self):
        return list(self.selection_functions.keys())

    @property
    def counts(self):
        """ return a dictionary of the current counts for each selection"""
        return dict(zip(self._selectors.keys(), self._counts))

    @property
    def counts_weighted(self):
        """ return a dictionary of the current weighted counts for each selection"""
        return dict(zip(self._selectors.keys(), self._counts_weighted))

    def to_table(self):
        from astropy.table import Table

        keys = list(self.counts.keys())
        vals = list(self.counts.values())
        vals_w = list(self.counts_weighted.values())
        return Table({"criteria": keys, "counts": vals, "counts_weighted": vals_w})

    def _repr_html_(self):
        return self.to_table()._repr_html_()

    def __call__(self, value, weight=1):
        """ test that value passes all cuts in """
        self._counts_total += 1
        result = np.array(list(map(lambda f: f(value), self._selectors.values())))
        self._counts += result.astype(int)
        self._counts_weighted += result.astype(int) * weight
        return result


class ImageDataChecker(DataChecker):
    """ for configuring image-wise data checks """

    pass


class EventDataChecker(DataChecker):
    """ for configuring event-wise data checks """

    pass


def expand_tel_list(tel_list, max_tels, index_map):
    """
    un-pack var-length list of tel_ids into 
    fixed-width bit pattern by tel_index
    """
    pattern = np.zeros(max_tels).astype(bool)
    pattern[tel_list] = 1
    return pattern


def create_tel_id_to_tel_index_transform(sub):
    """
    build a mapping of tel_id back to tel_index:
    (note this should be part of SubarrayDescription)
    """
    idx = np.zeros(max(sub.tel_indices) + 1)
    for key, val in sub.tel_indices.items():
        idx[key] = val

    # the final transform then needs the mapping and the number of telescopes
    return partial(expand_tel_list, max_tels=len(sub.tel) + 1, index_map=idx)


class ImageCleaner(Component):
    """
    Apply Image Cleaning    
    """

    method = CaselessStrEnum(
        ["tailcuts-standard", "tailcuts-mars"],
        help="Image Cleaning Method to use",
        default_value="tailcuts-standard",
    ).tag(config=True)

    picture_threshold_pe = Float(
        help="top-level threshold in photoelectrons", default_value=10.0
    ).tag(config=True)

    boundary_threshold_pe = Float(
        help="second-level threshold in photoelectrons", default_value=5.0
    ).tag(config=True)

    min_picture_neighbors = Int(
        help="Minimum number of neighbors above threshold to consider", default_value=2
    ).tag(config=True)

    def _apply_tailcuts_standard(self, geom, image):
        return tailcuts_clean(
            geom,
            image,
            picture_thresh=self.picture_threshold_pe,
            boundary_thresh=self.boundary_threshold_pe,
            min_number_picture_neighbors=self.min_picture_neighbors,
        )

    def __call__(self, geom: "CameraGeometry", image: np.ndarray):
        """ Apply image cleaning
        
        Parameters
        ----------
        geom : CameraGeometry
            geometry definition of the camera
        image : np.ndarray
            image pixel data corresponding to the camera geometry 
        
        Returns
        -------
        np.ndarray
            boolean mask of pixels passing cleaning
        """
        return self._apply_tailcuts_standard(geom, image)


class Stage1Process(Tool):
    name = "ctapipe-stage1-process"
    description = "process R0,R1,DL0 inputs into DL1 outputs"

    input_filename = Unicode(help="DL0 input filename").tag(config=True)
    output_filename = Unicode(
        help="DL1 output filename", default_value="dl1_events.h5"
    ).tag(config=True)
    write_images = Bool(
        help="Store DL1/Event/Image data in output", default_value=False
    ).tag(config=True)

    aliases = Dict(
        {
            "input": "Stage1Process.input_filename",
            "output": "Stage1Process.output_filename",
            "allowed-tels": "EventSource.allowed_tels",
            "max-events": "EventSource.max_events",
        }
    )

    flags = {
        "write-images": (
            {"Stage1Process": {"write_images": True}},
            "store DL1/Event images in output",
        )
    }

    classes = List([EventSource, CameraCalibrator, ImageCleaner, ImageDataChecker])

    def setup(self):

        if self.input_filename == "":
            raise ToolConfigurationError("Please specify --input <DL0/Events file>")

        self.event_source = self.add_component(
            EventSource.from_url(self.input_filename, parent=self)
        )

        self.calibrate = self.add_component(CameraCalibrator(parent=self))
        self.clean = self.add_component(ImageCleaner(parent=self))

        # TODO: eventually configure this from file
        self.check_image = self.add_component(
            ImageDataChecker(
                selection_functions=dict(
                    enough_pixels="lambda im: np.count_nonzero(im) > 2",
                    enough_charge="lambda im: im.sum() > 100",
                )
            )
        )

        # self.check_image_parameters = DataChecker(
        #     "ParameterSelection",
        #     OrderedDict(
        #         good_moments="lambda p: p.hillas.width >= 0 and p.hillas.length >= 0",
        #         min_ellipticity="lambda p: p.hillas.width/p.hillas.length > 0.1",
        #         max_ellipticity="lambda p: p.hillas.width/p.hillas.length < 0.6",
        #         nominal_distance="lambda p: True",  # TODO: implement
        #     ),
        # )

        # setup HDF5 compression:
        self._hdf5_filters = tables.Filters(
            complevel=5,  # enable compression, with level 0=disabled, 9=max
            complib="blosc:zstd",  #  compression using blosc
            fletcher32=True,  # attach a checksum to each chunk for error correction
        )

    def write_simulation_configuration(self, event, writer):
        """
        Write the simulation headers to a single row of a table. Later 
        if this file is merged with others, that table will grow. 

        Note that this function should be run first
        """
        self.log.debug("Writing simulation configuration")

        class ExtraMCInfo(Container):
            obs_id = Field(0, "MC Run Identifier")

        extramc = ExtraMCInfo()

        extramc.obs_id = event.dl0.obs_id
        writer.write("simulation/run_config", [extramc, event.mcheader])

    def write_simulation_histograms(self):
        self.log.debug("Writing simulation histograms")

    def write_instrument_configuration(self, subarray):
        """write the SubarrayDescription
        
        Parameters
        ----------
        subarray : ctapipe.instrument.SubarrayDescription
            subarray description 
        """
        self.log.debug("Writing instrument configuration")
        serialize_meta = True

        subarray.to_table().write(
            self.output_filename,
            path="/instrument/subarray/layout",
            serialize_meta=serialize_meta,
            append=True,
        )
        subarray.to_table(kind="optics").write(
            self.output_filename,
            path="/instrument/telescope/optics",
            append=True,
            serialize_meta=serialize_meta,
        )
        for telescope_type in subarray.telescope_types:
            ids = set(subarray.get_tel_ids_for_type(telescope_type))
            print(f"{telescope_type}: {len(ids)}")
            if len(ids) > 0:  # only write if there is a telescope with this camera
                tel_id = list(ids)[0]
                camera = subarray.tel[tel_id].camera
                camera.to_table().write(
                    self.output_filename,
                    path=f"/instrument/telescope/camera/{camera}",
                    append=True,
                    serialize_meta=serialize_meta,
                )

    def parameterize_image(
        self, telescope: TelescopeDescription, data: DL1CameraContainer
    ):
        """Apply Image Cleaning
       
        Parameters
        ----------
        telescope : TelescopeDescription
           telescope description
        data : DL1CameraContainer
            calibrated camera data
        
        Returns
        -------
        np.ndarray, ImageParametersContainer: 
            cleaning mask, parameters
        """

        # apply cleaning

        mask = self.clean(geom=telescope.camera, image=data.image)

        clean_image = data.image.copy()
        clean_image[~mask] = 0

        params = ImageParametersContainer()

        # check if image can be parameterized:

        image_criteria = self.check_image(clean_image)
        self.log.debug(
            "image_criteria: %s",
            list(zip(self.check_image.criteria_names, image_criteria)),
        )
        if all(image_criteria):
            # parameterize
            params.hillas = hillas_parameters(geom=telescope.camera, image=clean_image)
            params.timing = timing_parameters(
                geom=telescope.camera,
                image=data.image,
                pulse_time=data.pulse_time,
                hillas_parameters=params.hillas,
            )
            params.leakage = leakage(
                geom=telescope.camera, image=data.image, cleaning_mask=mask
            )
            params.concentration = concentration(
                geom=telescope.camera,
                image=clean_image,
                hillas_parameters=params.hillas,
            )

        return clean_image, params

    def write_events(self):
        self.log.debug("Writing DL1/Event data")
        tel_index = TelEventIndexContainer()
        event_index = EventIndexContainer()

        with HDF5TableWriter(
            self.output_filename,
            group_name="dl1",
            mode="a",
            add_prefix=True,
            filters=self._hdf5_filters,
        ) as writer:

            for event in self.event_source:
                self.log.debug("Writing event_id=%s", event.dl0.event_id)

                self.calibrate(event)

                event.mc.prefix = "mc"
                event.trig.prefix = ""
                event_index.event_id = event.dl0.event_id
                event_index.obs_id = event.dl0.obs_id
                tel_index.event_id = event.dl0.event_id
                tel_index.obs_id = event.dl0.obs_id

                # On the first event, we now have a subarray loaded, and other info, so
                # we can write the configuration data.
                if event.count == 0:
                    tel_list_transform = create_tel_id_to_tel_index_transform(
                        event.inst.subarray
                    )
                    writer.add_column_transform(
                        table_name="event/subarray/trigger",
                        col_name="tels_with_trigger",
                        transform=tel_list_transform,
                    )
                    self.write_simulation_configuration(event, writer)
                    self.write_instrument_configuration(event.inst.subarray)

                # write the subarray tables
                writer.write(
                    table_name="event/subarray/mc_shower",
                    containers=[event_index, event.mc],
                )
                writer.write(
                    table_name="event/subarray/trigger",
                    containers=[event_index, event.trig],
                )

                # write the telescope tables
                for tel_id, data in event.dl1.tel.items():

                    telescope = event.inst.subarray.tel[tel_id]
                    tel_type = str(telescope)
                    tel_index.tel_id = tel_id
                    tel_index.tel_type_id = hash(tel_type)

                    image_mask, params = self.parameterize_image(telescope, data)

                    self.log.debug("params: %s", params.as_dict(recursive=True))

                    if params.hillas.intensity is np.nan:
                        # TODO: dont' do this in the future! But right now, the HDF5TableWriter
                        # has problems if the first event has NaN as a value, since it can't
                        # infer the type.
                        continue

                    self.log.debug("Writing! ******")

                    containers_to_write = [
                        tel_index,
                        params.hillas,
                        params.timing,
                        params.leakage,
                        params.concentration,
                    ]

                    data.prefix = ""  # don't want a prefix for this container
                    # extra_im.tel_id = tel_id

                    if self.write_images:
                        writer.write(
                            table_name=f"event/telescope/image/{tel_type}",
                            containers=[tel_index, event.dl0, data],
                        )

                    writer.write(
                        table_name=f"event/telescope/parameters",
                        containers=containers_to_write,
                    )

    def generate_indices(self):
        import itertools

        with tables.open_file(self.input_filename, mode="a") as h5file:
            for node in itertools.chain(
                h5file.iter_nodes("/dl1/event/telescope"),
                h5file.iter_nodes("/dl1/event/subarray"),
            ):
                if isinstance(node, tables.group.Group):
                    pass
                else:
                    if "event_id" in node.colnames:
                        node.cols.event_id.create_index()
                        self.log.debug("generated event_id index")
                    if "tel_id" in node.colnames:
                        node.cols.tel_id.create_index()
                        self.log.debug("generated tel_id index")
                    if "obs_id" in node.colnames:
                        self.log.debug("generated obs_id index")
                        node.cols.obs_id.create_index(kind="ultralight")

    def start(self):

        self.write_events()
        self.generate_indices()
        self.write_simulation_histograms()


if __name__ == "__main__":
    tool = Stage1Process()
    tool.run()
