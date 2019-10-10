"""
NWBIO
========

IO class for reading data from a Neurodata Without Borders (NWB) dataset

Documentation : https://neurodatawithoutborders.github.io
Depends on: h5py, nwb, dateutil
Supported: Read, Write
Specification - https://github.com/NeurodataWithoutBorders/specification
Python APIs - (1) https://github.com/AllenInstitute/nwb-api/tree/master/ainwb 
	          (2) https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/core/nwb_data_set.py 
              (3) https://github.com/NeurodataWithoutBorders/api-python
Sample datasets from CRCNS - https://crcns.org/NWB
Sample datasets from Allen Institute - http://alleninstitute.github.io/AllenSDK/cell_types.html#neurodata-without-borders
"""

from __future__ import absolute_import

import logging
from os.path import join
import dateutil.parser
import numpy as np
import quantities as pq

try:
    import pynwb
except ImportError as err:
    HAVE_PYNWB = False
else:
    HAVE_PYNWB = True

from neo.core import (objectlist, Segment, SpikeTrain, Unit, Epoch, Event, AnalogSignal,
                      IrregularlySampledSignal, ChannelIndex, Block)
from neo.io.baseio import BaseIO
from neo.core.baseneo import MergeError

logger = logging.getLogger('Neo')

class NWBIO(BaseIO):
    """
    Class for reading experimental data from a NWB format file.

    Writing to NWB is not yet supported by this IO
    """

    supported_objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
                         SpikeTrain, Epoch, Event] # maybe to remove at the end : already declared in neo.core.objectlist
    readable_objects  = supported_objects
    writeable_objects = supported_objects

    has_header = False

    name = 'NeoNWB IO'
    description = 'This IO reads/writes experimental data from/to an .nwb dataset'
    extensions = ['nwb']
    mode = 'one-file'

    is_readable = True
    is_writable = True
    # is_streameable = False

    def __init__(self, filename, mode):
        """
        Arguments:
            filename : the filename
        """
        BaseIO.__init__(self, filename=filename)
        self.filename = filename
        if mode == "w":
            print("test write")
        else:
            io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
            self._file = io.read() # Define the file as a NWBFile object

    def read_all_blocks(self, lazy=False, merge_singles=True, **kargs):
        """
        Loads all blocks in the file that are attached to the root (which
        happens when they are saved with save() or write_block()).

        If `merge_singles` is True, then the IO will attempt to merge single channel
         `AnalogSignal` objects into multichannel objects, and similarly for single `Epoch`,
         `Event` and `IrregularlySampledSignal` objects.
        """
        assert not lazy, 'Do not support lazy'

        self.merge_singles = merge_singles

        blocks = []
        for node in self._file.acquisition:
        # for name, node in self._file.items():
            # print(name, node)
            # if "Block" in name:
            blocks.append(self._read_block(node))
        return blocks

    def read_block(self, lazy=False, **kargs):
        """
        Load the first block in the file.
        """
        assert not lazy, 'Do not support lazy'
        return self.read_all_blocks(lazy=lazy)[0]

    def _read_block(self, node, lazy=False, cascade=True, **kwargs):
        self._lazy = lazy
        file_access_dates = self._file.file_create_date
        identifier = self._file.identifier # or experimenter ?
        if identifier == '_neo':  # this is an automatically generated name used if block.name is None
            identifier = None
        description = self._file.session_description
        if description == "no description":
            description = None
        block = Block(name=identifier, 
                      description=description,
                      file_origin=self.filename,
                      file_datetime=file_access_dates,
                      rec_datetime=self._file.session_start_time,
                      file_access_dates=file_access_dates,
                      file_read_log='')
        if cascade:
            # self._handle_general_group(block)
            self._handle_epochs_group(node, lazy, block)
            # self._handle_acquisition_group(lazy, block)
            # self._handle_stimulus_group(lazy, block)
            # self._handle_processing_group(block)
            # self._handle_analysis_group(block)
        self._lazy = False
        return block

    def write_block(self, block, **kwargs):
        start_time = datetime.now()
        for i in self.filename:
            self._file = NWBFile(self.filename,            
                               session_start_time=start_time,
                               identifier=block.name or "_neo",
                               file_create_date=None,
                               timestamps_reference_time=None,
                               experimenter=None,
                               experiment_description=None,
                               session_id=None,
                               institution=None,
                               keywords=None,
                               notes=None,
                               pharmacology=None,
                               protocol=None,
                               related_publications=None,
                               slices=None,
                               source_script=None,
                               source_script_file_name=None,
                               data_collection=None,
                               surgery=None,
                               virus=None,
                               stimulus_notes=None,
                               lab=None,
                               acquisition=None,
                               stimulus=None,
                               stimulus_template=None,
                               epochs=None,
                               epoch_tags=set(),
                               trials=None,
                               invalid_times=None,
                               time_intervals=None,
                               units=None,
                               modules=None,
                               electrodes=None,
                               electrode_groups=None,
                               ic_electrodes=None,
                               sweep_table=None,
                               imaging_planes=None,
                               ogen_sites=None,
                               devices=None,
                               subject=None
                               )
            io_nwb = pynwb.NWBHDF5IO(self.filename, manager=get_manager(), mode='w')
            for segment in block.segments:
                self._write_segment(segment)
            io_nwb.write(self._file)
            io_nwb.close()

    def _handle_general_group(self, block):
        pass

    def _handle_epochs_group(self, node, lazy, block):
        # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
        self._lazy = lazy
        print(node)
        epochs = self._file.acquisition[node]
        print(self._file.acquisition[node])
        # for key in epochs:
        key = node
        timeseries = []
        current_shape = self._file.get_acquisition(key).data.shape[0] # sample number
        times = np.zeros(current_shape)
        for j in range(0, current_shape):
            print("in handle epoch",j)
            times[j]=1./self._file.get_acquisition(key).rate*j+self._file.get_acquisition(key).starting_time
            if times[j] == self._file.get_acquisition(key).starting_time:
                t_start = times[j] * pq.second
            elif times[j]==times[-1]:
                t_stop = times[j] * pq.second
            else:
                print(node)
                # timeseries.append(self._handle_timeseries(self._lazy, key, times[j]))
            segment = Segment(name=j)
#         for obj in timeseries:
#             obj.segment = segment
#             if isinstance(obj, AnalogSignal):
#                 #print("AnalogSignal")
#                 segment.analogsignals.append(obj)
#             elif isinstance(obj, IrregularlySampledSignal):
#                 #print("IrregularlySampledSignal")
#                 segment.irregularlysampledsignals.append(obj)
#             elif isinstance(obj, Event):
#                 #print("Event")
#                 segment.events.append(obj)
#             elif isinstance(obj, Epoch):
#                 #print("Epoch")
#                 segment.epochs.append(obj)
#         segment.block = block
#         segment.times=times

# #            print("segment.block = ", segment.block)
# #            print("block = ", block)
# #            print("segment = ", segment)
# #            print("segments = ", segments)
# #            block.segments.append(segment)
#         return segment, obj, times

    def _handle_timeseries(self,lazy, name, timeseries):
        for i in self._file.acquisition:
            data_group = self._file.get_acquisition(i).data*self._file.get_acquisition(i).conversion
            dtype = data_group.dtype
            if lazy==True:
                data = np.array((), dtype=dtype)
                lazy_shape = data_group.shape
            else:
                data = data_group

            if dtype.type is np.string_:
                if self._lazy:
                    times = np.array(())
                else:
                    times = self._file.get_acquisition(i).timestamps
                duration = 1/self._file.get_acquisition(i).rate
                if durations:
                    # Epoch
                    if self._lazy:
                        durations = np.array(())
                    obj = Epoch(times=times,
                                durations=durations,
                                labels=data_group,
                                units='second')
                else:
                    # Event
                    obj = Event(times=times,
                                labels=data_group,
                                units='second')
            else:
                units = self._file.get_acquisition(i).unit
            current_shape = self._file.get_acquisition(i).data.shape[0] # number of samples
            times = np.zeros(current_shape)
            for j in range(0, current_shape):
                print("in handle timseries", i, j)
                times[j]=1./self._file.get_acquisition(i).rate*j+self._file.get_acquisition(i).starting_time
                if times[j] == self._file.get_acquisition(i).starting_time:
                    sampling_metadata = times[j]
                    t_start = sampling_metadata * pq.s
                    sampling_rate = self._file.get_acquisition(i).rate * pq.Hz
                    obj = AnalogSignal(
                                       data_group,
                                       units=units,
                                       sampling_rate=sampling_rate,
                                       t_start=t_start,
                                       name=name)
                elif self._file.get_acquisition(i).timestamps:
                    if self._lazy:
                        time_data = np.array(())
                    else:
                        time_data = self._file.get_acquisition(i).timestamps
                    obj = IrregularlySampledSignal(
                                                data_group,
                                                units=units,
                                                time_units=pq.second)
            return obj

    def _handle_acquisition_group(self, lazy, block):
        acq = self._file.acquisition
# #                print("segment = ", segment)
    def _handle_stimulus_group(self, lazy, block):
        sti = self._file.stimulus
        for name in sti:
            segment_name_sti = self._file.epochs
            desc_sti = self._file.get_stimulus(name).unit
            segment_sti = segment_name_sti
            if lazy==True:
                times = np.array(())
                lazy_shape = self._file.get_stimulus(name).data.shape
            else:
                current_shape = self._file.get_stimulus(name).data.shape[0] # sample number
                times = np.zeros(current_shape)
                for j in range(0, current_shape): # For testing !
                    times[j]=1./self._file.get_stimulus(name).rate*j+self._file.get_acquisition(name).starting_time # times = 1./frequency [Hz] + t_start [s]
                spiketrain = SpikeTrain(times, units=pq.second,
                                         t_stop=times[-1]*pq.second)

    def _handle_processing_group(self, block):
        pass

    def _handle_analysis_group(self, block):
        pass

    def _write_segment(self, segment):
        start_time = segment.t_start
        stop_time = segment.t_stop

        nwb_epoch = self._file.add_epoch(
                                        self._file,
                                        segment.name,
                                        start_time=float(start_time),
                                        stop_time=float(stop_time),
                                        )

        for i, signal in enumerate(chain(segment.analogsignals, segment.irregularlysampledsignals)):
            self._write_signal(signal, nwb_epoch, i)
        self._write_spiketrains(segment.spiketrains, segment)
        for i, event in enumerate(segment.events):
            self._write_event(event, nwb_epoch, i)
        for i, neo_epoch in enumerate(segment.epochs):
            self._write_neo_epoch(neo_epoch, nwb_epoch, i)

    def _write_signal(self, signal, epoch, i):
        for i in self._file.acquisition:
            name = i
        signal_name = signal.name or "signal{0}".format(i)
        ts_name = "{0}".format(signal_name)

        # create a builder for the namespace
        ns_builder = NWBNamespaceBuilder("Extension for use in my laboratory", "mylab")

        # create extensions
        ts = NWBGroupSpec('A custom TimeSeries interface',
                            attributes=[],
                            datasets=[],
                            groups=[],
                            neurodata_type_inc='TimeSeries',
                            neurodata_type_def='MultiChannelTimeSeries')

        conversion = _decompose_unit(signal.units)
        attributes = {"conversion": conversion,
                      "resolution": float('nan')}

        if isinstance(signal, AnalogSignal):
            sampling_rate = signal.sampling_rate.rescale("Hz")
            signal.sampling_rate = sampling_rate

           # add the extension
            ext_source = 'nwb_neo_extension.specs.yaml'
            ts.add_dataset(
                            doc='',
                            neurodata_type_def='MultiChannelTimeSeries',
#                           ext_source, 
#                           "starting_time", 
#                           time_in_seconds(signal.t_start),
#                           {"rate": float(sampling_rate)},
                          )
        else:
            raise TypeError("signal has type {0}, should be AnalogSignal or IrregularlySampledSignal".format(signal.__class__.__name__))

    def _write_spiketrains(self, spiketrains, segment):
        mod = NWBGroupSpec('A custom TimeSeries interface',
                            attributes=[],
                            datasets=[],
                            groups=[],
                            neurodata_type_inc='TimeSeries',
                            neurodata_type_def='Module')

        ext_source = 'nwb_neo_extension.specs.yaml'
        mod.add_dataset(
                        doc='',
                        neurodata_type_def='Module',
                      )

    def _write_event(self, event, nwb_epoch, i):
        event_name = event.name or "event{0}".format(i)
        ts_name = "{0}_{1}".format(event.segment.name, event_name)
        ts = NWBGroupSpec('A custom TimeSeries interface',
                           attributes=[],
                           datasets=[],
                           groups=[],
                           neurodata_type_inc='TimeSeries',
                           neurodata_type_def='AnnotationSeries')

        ext_source = 'nwb_neo_extension.specs.yaml'
        mod.add_dataset(
                        doc='',
                        neurodata_type_def='AnnotationSeries',
                      )

        self._file.add_epoch_ts(
                               nwb_epoch,
                               time_in_seconds(event.segment.t_start),
                               time_in_seconds(event.segment.t_stop),
                               event_name,
                                )

    def _write_neo_epoch(self, neo_epoch, nwb_epoch, i):
        ts = NWBGroupSpec('A custom TimeSeries interface',
                            attributes=[],
                            datasets=[],
                            groups=[],
                            neurodata_type_inc='TimeSeries',
                            neurodata_type_def='AnnotatedIntervalSeries')
        ext_source = 'nwb_neo_extension.specs.yaml'
        mod.add_dataset(
                        doc='',
                        neurodata_type_def='AnnotatedIntervalSeries',
                      )

def time_in_seconds(t):
    return float(t.rescale("second"))

def _decompose_unit(unit):
    assert isinstance(unit, pq.quantity.Quantity)
    assert unit.magnitude == 1
    conversion = 1.0
    def _decompose(unit):
        dim = unit.dimensionality
        if len(dim) != 1:
            raise NotImplementedError("Compound units not yet supported")
        uq, n = dim.items()[0]
        if n != 1:
            raise NotImplementedError("Compound units not yet supported")
        uq_def = uq.definition
        return float(uq_def.magnitude), uq_def

prefix_map = {
    1e-3: 'milli',
    1e-6: 'micro',
    1e-9: 'nano'
}