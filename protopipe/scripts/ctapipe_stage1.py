""" 
Tool to process  {R0,R1,DL0}/Event data into DL1/Event data
"""
from collections import OrderedDict, namedtuple
from functools import partial
from pathlib import Path

import numpy as np
import tables.filters
import traitlets
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.core import Component, Container, Field, Tool, ToolConfigurationError
from ctapipe.core.traits import Bool, CaselessStrEnum, Dict, Float, Int, List, Unicode
from ctapipe.image import hillas_parameters, tailcuts_clean
from ctapipe.image.concentration import concentration
from ctapipe.image.leakage import leakage
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.instrument import CameraGeometry, TelescopeDescription
from ctapipe.io import EventSource, HDF5TableWriter
from ctapipe.io.containers import (
    DL1CameraContainer,
    EventIndexContainer,
    ImageParametersContainer,
    TelEventIndexContainer,
)
import tqdm

from ctapipe.core import Provenance

PROV = Provenance()
RangeTuple = namedtuple("RangeTuple", "min,max")


class TelescopeParameter(traitlets.Dict):
    """
    Apply a parameter to a telescope based on the telescope's id or type name.
    The various parameter values are given as a `dict` of string keys and float
    values. The keys can be:

    - '*' match all telescopes (default value)
    - a telescope type string (e.g. 'type SST_ASTRI_CHEC') to apply to all telescopes of
        that type
    - a glob pattern for a telescope string (e.g.  'type SST_*') to apply to all
        types matching that pattern
    - a specific telescope ID ('id 89')

    These are evaluated in-order, so you can first set a default value, and then set
    values for specific telescopes or types

    Examples
    --------

    .. code-block: python
    {
        '*': 5.0,                       # default
        'type LST_LST_LSTCam': 5.2,
        'type MST_MST_NectarCam': 4.0,
        'type MST_MST_FlashCam': 4.5,
        'id 34' 4.0,                   # override telescope 34 specifically
    }
    """

    def validate(self, obj, value):
        super().validate(obj, value)

        for key, val in value.items():
            if type(key) is not str:
                raise traitlets.TraitError("telescope type must be a string")
            if type(val) not in [float, int]:
                raise traitlets.TraitError(
                    f"value must be a float, got {val}" f" ({type(val)}) instead"
                )
            if not any(key.startswith(x) for x in ["*", "type ", "id ", "slice "]):
                raise traitlets.TraitError(
                    f"key '{key}' must be '*', 'type <type>', "
                    f"'id <tel_id>' or 'slice <slice>'"
                )
            if key.startswith("id "):
                tokens = key.split(" ")
                try:
                    tel_id = int(tokens[1])  # will fail if not parsable
                except ValueError as err:
                    raise traitlets.TraitError(
                        f"couldn't parse telescope id '{tokens[1]}'"
                    )

            value[key] = float(val)
        return value


class TelescopeParameterResolver:
    def __init__(
        self, subarray: "ctapipe.instrument.SubarrayDescription", tel_param: dict
    ):
        """
        Handles looking up a parameter by telescope_id, given a TelescopeParameter
        trait (which maps a parameter to a set of telescopes by type, id, or other
        selection criteria).

        Parameters
        ----------
        name: str
            name of the mapped parameter
        subarray: ctapipe.instrument.SubarrayDescription
            description of the subarray (includes mapping of tel_id to tel_type)
        tel_param: TelescopeParameter trait dict
            the parameter definitions as a map of key to value
        """

        # build dictionary mapping tel_id to parameter:
        self._value_for_tel_id = {}

        for key, value in tel_param.items():
            if key == "*":
                for tel_id in subarray.tel_ids:
                    self._value_for_tel_id[tel_id] = value
            else:
                command, argument = key.split(" ")
                if command == "type":
                    for tel_id in subarray.get_tel_ids_for_type(argument):
                        self._value_for_tel_id[tel_id] = value
                elif command == "id":
                    self._value_for_tel_id[int(argument)] = value
                else:
                    raise ValueError(f"Unrecognized command: {command}")

    def value_for_tel_id(self, tel_id: int) -> float:
        """
        returns the resolved parameter for the given telescope id
        """
        try:
            return self._value_for_tel_id[tel_id]
        except KeyError as err:
            raise KeyError(
                f"TelescopeParameterResolver: no "
                f"parameter value was set for telescope with tel_id="
                f"{tel_id}. Please set it explicity, or by telescope type or '*'."
            )


class DataChecker(Component):
    """
    Manages a set of selection criteria that operate on the same type of input.
    Each time it is called, it returns a boolean array of whether or not each
    criterion passed.
    """

    selection_functions = Dict(
        help=(
            "dict of '<cut name>' : lambda function in string format to accept ("
            "select) a given data value.  E.g. `{'mycut': 'lambda x: x > 3'}` "
        )
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

        # add a selection to count all entries and make it the first one
        selection_functions = OrderedDict(self.selection_functions)
        selection_functions["TOTAL"] = "lambda x: True"
        selection_functions.move_to_end("TOTAL", last=False)
        self.selection_functions = selection_functions  # update

        # generate real functions from the selection function strings
        self._selectors = OrderedDict(
            (name, eval(func_str)) for name, func_str in selection_functions.items()
        )

        # arrays for recording overall statistics
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
        return list(self.selection_functions.values())

    @property
    def counts(self):
        """ return a dictionary of the current counts for each selection"""
        return dict(zip(self._selectors.keys(), self._counts))

    @property
    def counts_weighted(self):
        """ return a dictionary of the current weighted counts for each selection"""
        return dict(zip(self._selectors.keys(), self._counts_weighted))

    def to_table(self, functions=False):
        """ Return a tabular view of the latest quality summary"""
        from astropy.table import Table

        keys = list(self.counts.keys())
        vals = list(self.counts.values())
        vals_w = list(self.counts_weighted.values())
        cols = {"criteria": keys, "counts": vals, "counts_weighted": vals_w}
        if functions:
            cols["func"] = self.selection_function_strings
        return Table(cols)

    def _repr_html_(self):
        return self.to_table()._repr_html_()

    def __call__(self, value, weight=1):
        """
        Test that value passes all cuts

        Parameters
        ----------
        value:
            the value to pass to each selection function
        weight: float
            a weight to apply for weighted statistics

        Returns
        -------
        np.ndarray:
            array of booleans with results of each selection criterion in order
        """
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
    pass


class TailcutsImageCleaner(ImageCleaner):
    """
    Apply Image Cleaning to a set of telescope images
    """

    picture_threshold_pe = TelescopeParameter(
        help="top-level threshold in photoelectrons", default_value={"*": 10.0}
    ).tag(config=True)

    boundary_threshold_pe = TelescopeParameter(
        help="second-level threshold in photoelectrons", default_value={"*": 5.0}
    ).tag(config=True)

    min_picture_neighbors = Int(
        help="Minimum number of neighbors above threshold to consider", default_value=2
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config, parent, **kwargs)
        self._pic_thresh_resolver = None
        self._bnd_thresh_resolver = None

    def _apply_tailcuts_standard(self, subarray, image, tel_id):

        if self._pic_thresh_resolver is None:
            self._pic_thresh_resolver = TelescopeParameterResolver(
                subarray, self.picture_threshold_pe
            )
            self._bnd_thresh_resolver = TelescopeParameterResolver(
                subarray, self.boundary_threshold_pe
            )

        return tailcuts_clean(
            subarray.tel[tel_id].camera,
            image,
            picture_thresh=self._pic_thresh_resolver.value_for_tel_id(tel_id),
            boundary_thresh=self._bnd_thresh_resolver.value_for_tel_id(tel_id),
            min_number_picture_neighbors=self.min_picture_neighbors,
        )

    def __call__(
        self,
        tel_id: int,
        subarray: "ctapipe.instrument.SubarrayDescription",
        image: np.ndarray,
    ):
        """ Apply image cleaning
        
        Parameters
        ----------
        geom : ctapipe.image.CameraGeometry
            geometry definition of the camera
        image : np.ndarray
            image pixel data corresponding to the camera geometry
        subarray: ctapipe.image.SubarrayDescription
            subarray definition (for mapping tel type to tel_id)
        tel_id:
            tel_id so we know which cut to use
        
        Returns
        -------
        np.ndarray
            boolean mask of pixels passing cleaning
        """
        return self._apply_tailcuts_standard(subarray, image, tel_id)


class Stage1Process(Tool):
    name = "ctapipe-stage1-process"
    description = "process R0,R1,DL0 inputs into DL1 outputs"

    output_filename = Unicode(
        help="DL1 output filename", default_value="dl1_events.h5"
    ).tag(config=True)

    write_images = Bool(
        help="Store DL1/Event/Image data in output", default_value=False
    ).tag(config=True)

    overwrite = Bool(help="overwrite output file if it exists").tag(config=True)

    aliases = {
        "input": "EventSource.input_url",
        "output": "Stage1Process.output_filename",
        "allowed-tels": "EventSource.allowed_tels",
        "max-events": "EventSource.max_events",
    }

    flags = {
        "write-images": (
            {"Stage1Process": {"write_images": True}},
            "store DL1/Event images in output",
        ),
        "overwrite": (
            {"Stage1Process": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
    }

    compression_level = Int(
        help="compression level, 0=None, 9=maximum", default_value=5, min=0, max=9
    ).tag(config=True)
    compression_type = CaselessStrEnum(
        values=["blosc:zstd", "zlib"],
        help="compressor algorithm to use. ",
        default_value="zlib",
    ).tag(config=True)

    classes = List([EventSource, CameraCalibrator, ImageCleaner, ImageDataChecker])

    def setup(self):

        # check output path:
        output_path = Path(self.output_filename).expanduser()
        if output_path.exists() and self.overwrite:
            self.log.warn(f"Overwriting {output_path}")
            output_path.unlink()
        PROV.add_output_file(str(output_path), role="DL1/Event")

        # setup components:

        self.event_source = self.add_component(EventSource.from_config(parent=self))
        self.calibrate = self.add_component(CameraCalibrator(parent=self))
        self.clean = self.add_component(TailcutsImageCleaner(parent=self))

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
            complevel=self.compression_level,
            complib=self.compression_type,
            fletcher32=True,  # attach a checksum to each chunk for error correction
        )

    def _write_simulation_configuration(self, writer, event):
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

    def _write_simulation_histograms(self, writer):
        self.log.debug("Writing simulation histograms")

    def _write_instrument_configuration(self, subarray):
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
            if len(ids) > 0:  # only write if there is a telescope with this camera
                tel_id = list(ids)[0]
                camera = subarray.tel[tel_id].camera
                camera.to_table().write(
                    self.output_filename,
                    path=f"/instrument/telescope/camera/{camera}",
                    append=True,
                    serialize_meta=serialize_meta,
                )

    def _parameterize_image(self, subarray, data, tel_id):
        """Apply Image Cleaning
       
        Parameters
        ----------
        subarray : SubarrayDescription
           subarray description
        data : DL1CameraContainer
            calibrated camera data
        tel_id: int
            which telescope is being cleaned
        
        Returns
        -------
        np.ndarray, ImageParametersContainer: 
            cleaning mask, parameters
        """

        tel = subarray.tel[tel_id]

        # apply cleaning

        mask = self.clean(subarray=subarray, image=data.image, tel_id=tel_id)

        clean_image = data.image.copy()
        clean_image[~mask] = 0

        params = ImageParametersContainer()

        # check if image can be parameterized:

        image_criteria = self.check_image(clean_image)
        self.log.debug(
            "image_criteria: %s",
            list(zip(self.check_image.criteria_names, image_criteria)),
        )

        # parameterize the event if all criteria pass:
        if all(image_criteria):
            params.hillas = hillas_parameters(geom=tel.camera, image=clean_image)
            params.timing = timing_parameters(
                geom=tel.camera,
                image=clean_image,
                pulse_time=data.pulse_time,
                hillas_parameters=params.hillas,
            )
            params.leakage = leakage(
                geom=tel.camera, image=data.image, cleaning_mask=mask
            )
            params.concentration = concentration(
                geom=tel.camera, image=clean_image, hillas_parameters=params.hillas
            )

        return clean_image, params

    def _process_events(self, writer):
        self.log.debug("Writing DL1/Event data")
        tel_index = TelEventIndexContainer()
        event_index = EventIndexContainer()

        for event in tqdm.tqdm(self.event_source, total=self.event_source.max_events):
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
                    table_name="dl1/event/subarray/trigger",
                    col_name="tels_with_trigger",
                    transform=tel_list_transform,
                )
                self.subarray = event.inst.subarray
                event.inst.subarray.info(printer=self.log.debug)
                self._write_simulation_configuration(writer, event)
                self._write_instrument_configuration(event.inst.subarray)

            # write the subarray tables
            writer.write(
                table_name="dl1/event/subarray/mc_shower",
                containers=[event_index, event.mc],
            )
            writer.write(
                table_name="dl1/event/subarray/trigger",
                containers=[event_index, event.trig],
            )
            # write the telescope tables
            self._write_telescope_event(writer, event, tel_index)

    def _write_telescope_event(self, writer, event, tel_index):
        """
        add entries to the event/telescope tables for each telescope in a single
        event
        """

        # write the telescope tables
        for tel_id, data in event.dl1.tel.items():

            telescope = event.inst.subarray.tel[tel_id]
            tel_type = str(telescope)
            tel_index.tel_id = tel_id
            tel_index.tel_type_id = hash(tel_type)

            image_mask, params = self._parameterize_image(
                event.inst.subarray, data, tel_id=tel_id
            )

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

            if self.write_images:
                writer.write(
                    table_name=f"dl1/event/telescope/image/{tel_type}",
                    containers=[tel_index, event.dl0, data],
                )

            writer.write(
                table_name=f"dl1/event/telescope/parameters",
                containers=containers_to_write,
            )

    def _generate_indices(self, writer: HDF5TableWriter):
        import itertools

        h5file = writer._h5file

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

        with HDF5TableWriter(
            self.output_filename, mode="a", add_prefix=True, filters=self._hdf5_filters
        ) as writer:

            self._process_events(writer)
            self._write_simulation_histograms(writer)
            # self._generate_indices(writer)

    def finish(self):
        pass


if __name__ == "__main__":
    tool = Stage1Process()
    tool.run()
