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
from ctapipe.core import (Component, Container, Field, Tool,
                          ToolConfigurationError)
from ctapipe.core.traits import (Bool, CaselessStrEnum, Dict, Float, Int, List,
                                 Unicode)
from ctapipe.image import (HillasParameterizationError, hillas_parameters,
                           tailcuts_clean)
from ctapipe.image.concentration import concentration
from ctapipe.image.leakage import leakage
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.instrument import CameraGeometry, TelescopeDescription
from ctapipe.io import EventSource, HDF5TableWriter
from ctapipe.io.containers import (ConcentrationContainer, DL1CameraContainer,
                                   EventIndexContainer,
                                   HillasParametersContainer,
                                   ImageParametersContainer, LeakageContainer,
                                   TelEventIndexContainer,
                                   TimingParametersContainer)
from ctapipe.utils import CutFlow

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


class EventSelector:
    def __init__(self, name: str, selector_dict: OrderedDict):
        """
        Manages a set of selection cuts on the same input quantity or class.

        Allows one to access the current set of cuts as a Container  or a boolean value
        (so the results can be output per event if necessary)
        
        Parameters
        ----------
        name : str
            name of this set of cuts
        selector_dict: OrderedDict[str, str]
            dictionary of "cut name" to acceptance function string. 
            The acceptance functions are string represention of a lambda function 
            and return True if the input value is *accepted* 
        """
        self.name = name
        self._selectors = OrderedDict(
            (name, eval(func_str)) for name, func_str in selector_dict.items()
        )
        self.selection_function_strs = list(selector_dict.keys())

        self.event_count = 0
        self._counts = np.zeros(len(self._selectors), dtype=np.int)
        self._counts_weighted = np.zeros(len(self._selectors), dtype=np.int)

        # generate a Container that we can use for output somehow...
        self._container = Container()

    def __call__(self, value, weight=1):
        """ test that value passes all cuts in """
        self.event_count += 1
        result = np.array(list(map(lambda f: f(value), self._selectors.values())))
        self._counts += result.astype(int)
        self._counts_weighted += result.astype(int) * weight
        return result


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
    return partial(
        expand_tel_list, max_tels=len(sub.tel) + 1, index_map=idx
    )


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


class ImageSelection(Component):
    charge = Range(help="allowed charge in PE", default_value=(50, np.inf))
    num_pixels = Range(
        help="allowed number of pixels in image", default_value=(2, np.inf)
    )


class Stage1Process(Tool):
    name = "ctapipe-stage1-process"
    description = "process R0,R1,DL0 inputs into DL1 outputs"

    input_filename = Unicode(help="DL0 input filename").tag(config=True)
    output_filename = Unicode(
        help="DL1 output filename", default_value="dl1_events.h5"
    ).tag(config=True)
    write_images = Bool(
        help="Store DL1/Event/Image data in output", default_value=False
    )

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

    classes = List([EventSource, CameraCalibrator, ImageCleaner, ImageSelection])

    def setup(self):
        self.image_selection = self.add_component(ImageSelection(parent=self))
        if self.input_filename == "":
            raise ToolConfigurationError("Please specify --input <DL0/Events file>")

        self.event_source = self.add_component(
            EventSource.from_url(self.input_filename, parent=self)
        )

        self.calibrate = self.add_component(CameraCalibrator(parent=self))
        self.clean = self.add_component(ImageCleaner(parent=self))

        # TODO: eventually configure this from file
        self.check_image = EventSelector(
            "ImageSelection",
            OrderedDict(
                enough_pixels="lambda im: np.count_nonzero(im) > 2",
                enough_charge="lambda im: im.sum() > 100",
            ),
        )

        self.check_image_parameters = EventSelector(
            "ParameterSelection",
            OrderedDict(
                good_moments="lambda p: p.hillas.width >= 0 and p.hillas.length >= 0",
                min_ellipticity="lambda p: p.hillas.width/p.hillas.length > 0.1",
                max_ellipticity="lambda p: p.hillas.width/p.hillas.length < 0.6",
                nominal_distance="lambda p: True",  # TODO: implement
            ),
        )

    def write_simulation_configuration(self):
        self.log.debug("Writing simulation configuration")

    def write_simulation_histograms(self):
        self.log.debug("Writing simulation histograms")

    def write_instrument_configuration(self):
        self.log.debug("Writing instrument configuration")

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
        self.log.debug(zip(self.check_image._selectors.keys(), image_criteria))
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
            self.output_filename, group_name="dl1", mode="a", add_prefix=True
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

                if event.count == 0:
                    tel_list_transform = create_tel_id_to_tel_index_transform(
                        event.inst.subarray
                    )
                    writer.add_column_transform(
                        table_name="subarray/trigger",
                        col_name="tels_with_trigger",
                        transform=tel_list_transform,
                    )

                # write sub tables
                writer.write(
                    table_name="subarray/mc_shower", containers=[event_index, event.mc]
                )
                writer.write(
                    table_name="subarray/trigger", containers=[event_index, event.trig]
                )

                # write tel tables
                for tel_id, data in event.dl1.tel.items():

                    tel_index.tel_id = tel_id
                    telescope = event.inst.subarray.tel[tel_id]
                    image_mask, params = self.parameterize_image(telescope, data)

                    self.log.debug("params: %s", params.as_dict(recursive=True))

                    if params.hillas.intensity is nan:
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
                    tel_type = str(telescope)

                    if self.write_images:
                        writer.write(
                            table_name=f"telescope/image/{tel_type}",
                            containers=[tel_index, event.dl0, data],
                        )

                    writer.write(
                        table_name=f"telescope/parameters/{tel_type}",
                        containers=containers_to_write,
                    )

    def generate_indices(self):
        pass

    def start(self):

        self.write_simulation_configuration()
        self.write_simulation_histograms()
        self.write_instrument_configuration()
        self.write_events()
        self.generate_indices()


if __name__ == "__main__":
    tool = Stage1Process()
    tool.run()
