""" 
User Tools to process  {R0,R1,DL0}/Event data into DL1/Event data
"""
import ctapipe
from collections import OrderedDict
from ctapipe.core import Tool, ToolConfigurationError, Container, Field
from ctapipe.core.traits import Bool, Dict, Float, Int, List, Unicode
from ctapipe.io import EventSource, HDF5TableWriter
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.image import tailcuts_clean, hillas_parameters, HillasParameterizationError
from ctapipe.image.timing_parameters import timing_parameters
from pathlib import Path
from ctapipe.io.containers import DL1CameraContainer
from ctapipe.instrument import TelescopeDescription
from ctapipe.utils import CutFlow


def expand_tel_list(tel_list, max_tels, index_map):
    """
    un-pack var-length list of tel_ids into 
    fixed-width bit pattern by tel_index
    """
    pattern = np.zeros(max_tels).astype(bool)
    pattern[tel_list] = 1
    return pattern


class ExtraImageInfo(Container):
    """ attach the tel_id """

    tel_id = Field(0, "Telescope ID")


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
        }
    )

    flags = {
        "write-images": (
            {"Stage1Process": {"write_images": True}},
            "store DL1/Event images in output",
        )
    }

    classes = List([EventSource, CameraCalibrator])

    def setup(self):
        if self.input_filename == "":
            raise ToolConfigurationError("Please specify --input <DL0/Events file>")

        self.event_source = self.add_component(
            EventSource.from_url(self.input_filename, parent=self)
        )

        self.calibrate = self.add_component(CameraCalibrator(parent=self))

    def write_simulation_configuration(self):
        self.log.debug("Writing simulation configuration")

    def write_simulation_histograms(self):
        self.log.debug("Writing simulation histograms")

    def write_instrument_configuration(self):
        self.log.debug("Writing instrument configuration")

    def parameterize_image(self, cuts: CutFlow, telescope: TelescopeDescription, data: DL1CameraContainer):
        """Apply Image Cleaning
       
        Parameters
        ----------
        cuts: CutFlow
            image cuts to apply
        telescope : TelescopeDescription
           telescope description
        data : DL1CameraContainer
            calibrated camera data
        
        Returns
        -------
        List[Container]: 
            list of parameter containers
        """

        # apply cleaning

        mask = tailcuts_clean(
            geom=telescope.camera,
            image=data.image,
            picture_thresh=10.0,
            boundary_thresh=5.0,
        )

        clean_image = data.image.copy()
        clean_image[~mask] = 0

        # parameterize

        hillas = hillas_parameters(telescope.camera, data.image)
        timing = timing_parameters(telescope.camera, data.pulse_time, hillas_parameters)

        return [hillas, timing]

    def write_events(self):
        self.log.debug("Writing DL1/Event data")

        cuts = CutFlow("ImageCuts")
        cuts.set_cuts(OrderedDict(
            no_cuts=None,
            min_pixel=lambda im: np.count_nonzero(im) < 3,
            min_charge=lambda im: im.sum() < 100,
        ))
        
        with HDF5TableWriter(self.output_filename, mode="a", add_prefix=True) as writer:

            for event in self.event_source:
                self.log.debug("Writing event_id=%s", event.dl0.event_id)
                self.calibrate(event)

                event.dl0.prefix = ""
                event.mc.prefix = "mc"
                event.trig.prefix = ""

                # write sub tables
                writer.write(
                    table_name="subarray/mc_shower", containers=[event.dl0, event.mc]
                )
                # writer.write(
                #     table_name="subarray/trigger", containers=[event.dl0, event.trig]
                # )

                # write tel tables
                for tel_id, data in event.dl1.tel.items():
                    telescope = event.inst.subarray.tel[tel_id]
                    params = self.parameterize_image(cuts, telescope, data)

                    tel.prefix = ""  # don't want a prefix for this container
                    # extra_im.tel_id = tel_id
                    tel_name = str(telescope)
                    if self.write_images:
                        writer.write(
                            table_name=f"telescope/image/{tel_name}",
                            containers=[event.dl0, tel],
                        )
                    
                    writer.write(
                        table_name='telescope/parameters',
                        containers = [event.dl0,] + params
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
