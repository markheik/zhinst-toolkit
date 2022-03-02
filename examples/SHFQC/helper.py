import numpy as np
import time
import numpy as np
import typing as t
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from zhinst.deviceutils.shfqa import SHFQA_SAMPLING_FREQUENCY
from zhinst.toolkit.driver.devices import SHFQA
from zhinst.toolkit import CommandTable, Waveforms, SHFQAChannelMode, AveragingMode
import copy
import inspect


def voltage_to_power_dBm(voltage):
    power = 10 * np.log10(np.abs(voltage) ** 2 / 50 * 1000)
    return power


def voltage_to_phase(voltage):
    phase = np.unwrap(np.angle(voltage))
    return phase

NumpyArray = t.TypeVar("NumpyArray")


def measure_resonator_pulse_with_scope(
    device: SHFQA,
    channel: int,
    trigger_input: str,
    pulse_length_seconds: float,
    envelope_delay: float = 0,
    margin_seconds: float = 400e-9,
    num_avg: int = 1,
    single: bool = True,
) -> NumpyArray:
    """Measure the pulse used for probing the resonator in pulsed spectroscopy.

    NOTE:
        This only works if the device under test transmits the signal at the
        selected center frequency. To obtain the actual pulse shape, the user
        must connect the output of the SHFQA back to the input on the same
        channel with a loopback cable.

    Args:
        device: shfqc device
        channel: Index of the SHFQA channel
        trigger_input: A string for selecting the trigger input
            Examples: "channel0_trigger_input0" or "software_trigger0"
        pulse_length_seconds: Length of the pulse to be measured in seconds
        envelope_delay: Time after the trigger that the OUT signal starts
            propagating
        margin_seconds: Margin to add to the pulse length for the total length
            of the scope trace.

    Returns:
      Complex samples of the measured scope trace.
    """

    scope_channel = 0
    print("Measure the generated pulse with the SHFQA scope.")
    segment_duration = pulse_length_seconds + margin_seconds
    device.scopes[0].configure(
        input_select={scope_channel: f"channel{channel}_signal_input"},
        num_samples=int(segment_duration * SHFQA_SAMPLING_FREQUENCY),
        trigger_input=trigger_input,
        num_segments=1,
        num_averages=num_avg,
        trigger_delay=envelope_delay,
    )
    print("Measure the generated pulse with the SHFQA scope.")
    print(
        f"NOTE: Envelope delay ({envelope_delay *1e9:.0f} ns) is used as scope "
        "trigger delay"
    )
    device.scopes[0].run(single=single)

    if trigger_input == "software_trigger0":
        # issue num_avg triggers
        for _ in range(num_avg):
            device.system.swtriggers[0].single(1)

    scope_trace, *_ = device.scopes[0].read(timeout=5)
    return scope_trace[scope_channel]


def set_trigger_loopback(session, device, rate=1000):
    """
    Start a continuous trigger pulse from marker 1 A using the internal loopback to trigger in 1 A
    """

    m_ch = 0
    low_trig = 2
    continuous_trig = 1
    daq = session.daq_server
    daq.syncSetInt(f"/{device.serial}/raw/markers/*/testsource", low_trig)
    daq.setInt(f"/{device.serial}/raw/markers/{m_ch}/testsource", continuous_trig)
    daq.setDouble(f"/{device.serial}/raw/markers/{m_ch}/frequency", rate)
    daq.setInt(f"/{device.serial}/raw/triggers/{m_ch}/loopback", 1)
    time.sleep(0.2)


def clear_trigger_loopback(session, device):
    m_ch = 0
    session.daq_server.setInt(f"/{device.serial}/raw/markers/*/testsource", 0)
    session.daq_server.setInt(f"/{device.serial}/raw/triggers/{m_ch}/loopback", 0)


def generate_flat_top_gaussian(
    frequencies, pulse_duration, rise_fall_time, sampling_rate, scaling=0.9
):
    """Returns complex flat top Gaussian waveforms modulated with the given frequencies.

    Args:

        frequencies (array): array specifying the modulation frequencies applied to each
                             output wave Gaussian

        pulse_duration (float): total duration of each Gaussian in seconds

        rise_fall_time (float): rise-fall time of each Gaussian edge in seconds

        sampling_rate (float): sampling rate in samples per second based on which to
                               generate the waveforms

        scaling (optional float): scaling factor applied to the generated waveforms (<=1);
                                  use a scaling factor <= 0.9 to avoid overshoots

    Returns:

        pulses (dict): dictionary containing the flat top Gaussians as values

    """
    if scaling > 1:
        raise ValueError(
            "The scaling factor has to be <= 1 to ensure the generated waveforms lie within the \
                unit circle."
        )

    from scipy.signal import gaussian

    rise_fall_len = int(rise_fall_time * sampling_rate)
    pulse_len = int(pulse_duration * sampling_rate)

    std_dev = rise_fall_len // 10

    gauss = gaussian(2 * rise_fall_len, std_dev)
    flat_top_gaussian = np.ones(pulse_len)
    flat_top_gaussian[0:rise_fall_len] = gauss[0:rise_fall_len]
    flat_top_gaussian[-rise_fall_len:] = gauss[-rise_fall_len:]

    flat_top_gaussian *= scaling

    pulses = {}
    time_vec = np.linspace(0, pulse_duration, pulse_len)

    for i, f in enumerate(frequencies):
        pulses[i] = flat_top_gaussian * np.exp(2j * np.pi * f * time_vec)

    return pulses


def plot_resonator_pulse_scope_trace(scope_trace):
    """Plots the scope trace measured from the function `measure_resonator_pulse_with_scope`.

    Args:

        scope_trace (array): array containing the complex samples of the measured scope trace

    """

    import matplotlib.pyplot as plt

    time_ticks_us = 1.0e6 * np.array(range(len(scope_trace))) / SHFQA_SAMPLING_FREQUENCY
    plt.plot(time_ticks_us, np.real(scope_trace))
    plt.plot(time_ticks_us, np.imag(scope_trace))
    plt.title("Resonator probe pulse")
    plt.xlabel(r"t [$\mu$s]")
    plt.ylabel("scope samples [V]")
    plt.legend(["real", "imag"])
    plt.grid()
    plt.show()


def generate_integration_weights(readout_pulses, rotation_angle=0):
    """Generates a dictionary with integration weights calculated as conjugates of the rotated
       readout pulses.

    Args:

        readout_pulses (dict): dictionary with items {waveform_slot : readout_pulse(array)}

        rotation_angle (optional float): angle by which to rotate the input pulses

    Returns:

        weights (dict): dictionary with items {waveform_slot: weights (array)}

    """

    weights = {}
    for waveform_slot, pulse in readout_pulses.items():
        weights[waveform_slot] = np.conj(pulse * np.exp(1j * rotation_angle))

    return weights


def plot_readout_results(data):
    """Plots the result data of a qubit readout as dots in the complex plane.

    Args:

      data (array): complex data to plot.

    """

    import matplotlib.pyplot as plt

    max_value = 0

    plt.rcParams["figure.figsize"] = [10, 10]

    for complex_number in data:
        real = np.real(complex_number)
        imag = np.imag(complex_number)

        plt.plot(real, imag, "x")

        max_value = max(max_value, max(abs(real)))
        max_value = max(max_value, max(abs(imag)))

    # zoom so that the origin is in the middle
    max_value *= 1.05
    plt.xlim([-max_value, max_value])
    plt.ylim([-max_value, max_value])

    plt.legend(range(len(data)))
    plt.title("qubit readout results")
    plt.xlabel("real part")
    plt.ylabel("imaginary part")
    plt.grid()
    plt.show()


def fit_data(
    x_data,
    y_data,
    function,
    start_param,
    do_plot,
    figsize=10,
    font=10,
    qubit=0,
    x_label="",
    y_label="",
    saveloc="",
):

    pars, cov = curve_fit(
        f=function, xdata=x_data, ydata=y_data, p0=start_param, bounds=(-np.inf, np.inf)
    )
    stdevs = np.sqrt(np.diag(cov))
    residuals = y_data - function(x_data, *pars)

    # plot it
    if do_plot:
        fig4, axs = plt.subplots(2, 1, figsize=figsize)
        fig4.suptitle(f"Qubit {qubit}", fontsize=font)
        axs[0].scatter(x_data, y_data, s=20, color="#00b3b3", label="Data")
        axs[0].plot(
            x_data, function(x_data, *pars), linestyle="--", linewidth=2, color="black"
        )
        axs[0].legend(loc=2, prop={"size": font})

        axs[0].set_xlabel(x_label, fontsize=font)
        axs[0].set_ylabel(y_label, fontsize=font)
        axs[0].tick_params(axis="both", which="major", labelsize=font)

        axs[1].plot(x_data, residuals, ".")
        axs[1].set_xlabel(x_label, fontsize=font)
        axs[1].set_ylabel("residuals", fontsize=font)
        axs[1].tick_params(axis="both", which="major", labelsize=font)

        plt.savefig(saveloc)
        plt.show()

    return pars, stdevs


def qc_wait_for_state_change(daq, node, value, timeout=1.0, sleep_time=0.005):
    start_time = time.time()
    while start_time + timeout >= time.time() and daq.getInt(node) != value:
        time.sleep(sleep_time)
    if daq.getInt(node) != value:
        raise TimeoutError(
            f"{node} did not change to expected value {value} within "
            f"{timeout} seconds."
        )


def init_parameters(par, number_of_qubits):
    for i in range(number_of_qubits):
        par["qubit_readout_frequencies"].append(0)
        par["qubit_readout_widths"].append(0)
        par["qubit_readout_amplitudes"].append(0)
        par["max_drive_strength"].append(0)
        par["qubit_T1_time"].append(0)
        par["qubit_T2_time"].append(0)
        par["qubit_single_gate_time"].append(0)
        par["qubit_drive_frequency"].append(0)
        par["qubit_pi_amplitude"].append(0)
        par["qubit_pi/2_amplitude"].append(0)
        par["qubit_readout_rotation"].append(0)
        par["qubit_qscale"].append(-1)


def enable_qa_channel(device, value):
    device.qachannels[0].input.on(value)
    device.qachannels[0].output.on(value)


def plot_spectroscopy(
    sweeper, wide_resonator_spectrum, fontsize, figsize, slope, saveloc
):
    xaxis = sweeper.get_offset_freq_vector() / 10 ** 6
    fig1, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].plot(xaxis, voltage_to_power_dBm(wide_resonator_spectrum["vector"]))
    axs[0].set_xlabel("frequency [MHz]", fontsize=fontsize)
    axs[0].set_ylabel("amplitude [dBm]", fontsize=fontsize)
    axs[0].tick_params(axis="both", which="major", labelsize=fontsize)

    axs[1].plot(
        xaxis, voltage_to_phase(wide_resonator_spectrum["vector"]) + xaxis * slope
    )
    axs[1].set_xlabel("frequency [MHz]", fontsize=fontsize)
    axs[1].set_ylabel("phase [rad]", fontsize=fontsize)
    axs[1].tick_params(axis="both", which="major", labelsize=fontsize)

    plt.savefig(saveloc)

    plt.show()

    return fig1


def plot_spectroscopy_multiQubit(
    num_points,
    number_of_qubits,
    resonator_spectrum_data,
    relative_amplitude_values,
    fontsize,
    fontsize2,
    figsize,
    slope,
    saveloc,
):
    for qubit in range(number_of_qubits):
        number_amplitude_values = np.size(relative_amplitude_values)
        x_data = np.zeros((number_amplitude_values, num_points))
        y_data = np.zeros((number_amplitude_values, num_points))
        z_data = np.zeros((number_amplitude_values, num_points), dtype=complex)
        slope_array = np.zeros((number_amplitude_values, num_points))

        for amp_ind, amplitude in enumerate(relative_amplitude_values):
            spec_path = resonator_spectrum_data["qubits"][qubit][amp_ind]
            spec_path_props = spec_path["properties"]

            x_data[amp_ind] = (
                np.linspace(
                    spec_path_props["startfreq"],
                    spec_path_props["stopfreq"],
                    spec_path_props["numpoints"],
                )
                / 10 ** 6
            )
            y_data[amp_ind] = amplitude
            z_data[amp_ind] = spec_path["vector"]
            slope_array[amp_ind] = slope * x_data[amp_ind]

        fig2, axs = plt.subplots(1, 2, figsize=figsize)
        fig2.suptitle(f"Qubit {qubit}", fontsize=fontsize)
        color_amp = axs[0].scatter(
            x_data.flatten(),
            y_data.flatten(),
            s=20000,
            c=voltage_to_power_dBm(z_data.flatten()),
            marker="s",
        )
        axs[0].set_ylim([0, np.max(relative_amplitude_values)])
        axs[0].set_title("amplitude [dBm]", fontsize=fontsize2)
        axs[0].set_xlabel("frequency [MHz]", fontsize=fontsize2)
        axs[0].set_ylabel("rel. probe amplitude", fontsize=fontsize2)
        axs[0].tick_params(axis="both", which="major", labelsize=fontsize2)
        cbar = fig2.colorbar(color_amp, ax=axs[0])
        cbar.ax.tick_params(labelsize=fontsize2)

        color_phase = axs[1].scatter(
            x_data.flatten(),
            y_data.flatten(),
            s=20000,
            c=np.unwrap(np.angle(z_data)).flatten() + slope_array.flatten(),
            marker="s",
        )
        axs[1].set_ylim([0, np.max(relative_amplitude_values)])
        axs[1].set_title("phase [rad]", fontsize=fontsize2)
        axs[1].set_xlabel("frequency [MHz]", fontsize=fontsize2)
        axs[1].set_ylabel("rel. probe amplitude", fontsize=fontsize2)
        axs[1].tick_params(axis="both", which="major", labelsize=fontsize2)
        cbar = fig2.colorbar(color_phase, ax=axs[1])
        cbar.ax.tick_params(labelsize=fontsize2)

        plt.savefig(saveloc + f"{qubit}.png")
        plt.show()


def take_propagation_delay_measurement(
    session, device, qubit_params, num_avg, flat_top_gaussians, pulse_length_seconds
):
    spectroscopy_delay = qubit_params["propagation_delay"]
    pulse_envelope = flat_top_gaussians[0]
    envelope_delay = 0e-9

    # trigger_source = "channel0_trigger_input0"
    # trigger_source = "software_trigger0"
    # enable the loopback trigger
    set_trigger_loopback(session, device, rate=500e3)

    # generate the complex pulse envelope with a modulation frequency at the readoutfrequency of qubit
    with device.set_transaction():
        device.qachannels[0].spectroscopy.envelope.delay(0)
        device.qachannels[0].spectroscopy.delay(spectroscopy_delay)
        device.qachannels[0].spectroscopy.envelope.wave(pulse_envelope)
        device.qachannels[0].input.on(1)
        device.qachannels[0].output.on(1)
        device.qachannels[0].oscs[0].freq(0)
        device.qachannels[0].spectroscopy.envelope.enable(1)
        device.qachannels[0].spectroscopy.trigger.channel(0)

    scope_trace = measure_resonator_pulse_with_scope(
        device,
        channel=0,
        trigger_input=0,  # trigger_source,
        pulse_length_seconds=pulse_length_seconds,
        envelope_delay=envelope_delay,
        num_avg=num_avg,
        single=True,
    )

    clear_trigger_loopback(session, device)
    enable_qa_channel(device, 0)

    return scope_trace


def determine_propagation_delay(scope_trace, flat_top_gaussians, qubit_params):
    # simple filter window size for filtering the scope data
    window_s = 4
    pulse_envelope = flat_top_gaussians[0]
    envelope_delay = 0e-9
    spectroscopy_delay = qubit_params["propagation_delay"]

    # apply the filter to both the intial pulse and the one observed with the scope
    # to introduce the same amount of 'delay'
    pulse_smooth = np.convolve(
        np.abs(pulse_envelope), np.ones(window_s) / window_s, mode="same"
    )
    pulse_diff = np.diff(np.abs(pulse_smooth))
    sync_tick = np.argmax(pulse_diff)

    scope_smooth = np.diff(np.abs(scope_trace))
    scope_diff = np.convolve(scope_smooth, np.ones(window_s) / window_s, mode="same")
    sync_tack = np.argmax(scope_diff)

    delay_in_ns = 1.0e9 * (sync_tack - sync_tick) / SHFQA_SAMPLING_FREQUENCY
    delay_in_ns = 2 * ((delay_in_ns + 1) // 2)  # round up to the 2ns resolution
    print(f"delay between generator and monitor: {delay_in_ns} ns")
    print(f"Envelope delay: {envelope_delay * 1e9:.0f} ns")
    if spectroscopy_delay * 1e9 == delay_in_ns + (envelope_delay * 1e9):
        print("Spectroscopy delay and envelope perfectly timed!")
    else:
        print(
            f"Consider setting the spectroscopy delay to [{(envelope_delay + (delay_in_ns * 1e-9))}] "
        )
        print("to exactly integrate the envelope.")


def configure_device(
    device,
    number_of_qubits,
    qachannel_center_frequency,
    qachannel_power_in,
    qachannel_power_out,
    sgchannel_number,
    sgchannel_center_frequency,
    sgchannel_power_out,
    sgchannel_trigger_input,
):

    # configure inputs and outputs
    device.qachannels[0].configure_channel(
        center_frequency=qachannel_center_frequency,
        input_range=qachannel_power_in,
        output_range=qachannel_power_out,
        mode=SHFQAChannelMode.READOUT,
    )

    # configure sg channels
    with device.set_transaction():
        device.qachannels[0].markers[0].source("channel0_sequencer_trigger0")
        device.qachannels[0].markers[1].source("channel0_sequencer_trigger0")
        device.qachannels[0].generator.auxtriggers[1].channel("channel0_trigger_input0")
        device.qachannels[0].triggers[0].level(0.1)
        for qubit in range(number_of_qubits):
            sg_channel = sgchannel_number[qubit]
            synth_channel = int(np.floor(sg_channel / 2)) + 1
            device.synthesizers[synth_channel].centerfreq(
                sgchannel_center_frequency[qubit]
            )
            device.sgchannels[sg_channel].output.range(sgchannel_power_out[qubit])
            device.sgchannels[sg_channel + 2].trigger.level(0.1)
            device.sgchannels[sg_channel].awg.auxtriggers[0].channel(
                sgchannel_trigger_input + 2
            )
            device.sgchannels[sg_channel].awg.auxtriggers[0].slope(1)
            device.sgchannels[sg_channel].marker.source("awg_trigger0")


def upload_readout_waveforms(
    device, readout_pulses, weights, qubit_params, number_of_qubits
):

    readout_pulses_scaled = copy.copy(readout_pulses)
    for qubit in range(number_of_qubits):
        readout_pulses_scaled[qubit] = (
            readout_pulses[qubit] * qubit_params["qubit_readout_amplitudes"][qubit]
        )

    device.qachannels[0].generator.write_to_waveform_memory(readout_pulses_scaled)

    # configure result logger and weighted integration
    device.qachannels[0].readout.write_integration_weights(
        weights,
        # compensation for the delay between generator output and input of the integration unit
        integration_delay=qubit_params["propagation_delay"],
    )


def prepare_sequencer_and_resultLogger(
    device,
    integration_time_qubit_spectroscopy,
    readout_pulse_duration_qubit_spectroscopy,
    num_sweep_steps_qubit_spectroscopy,
    number_of_qubits,
):

    averages_per_sweep_step = int(
        np.round(
            integration_time_qubit_spectroscopy
            / readout_pulse_duration_qubit_spectroscopy
        )
    )
    device.qachannels[0].readout.configure_result_logger(
        result_length=num_sweep_steps_qubit_spectroscopy,
        num_averages=averages_per_sweep_step,
        result_source="integration",
        averaging_mode=AveragingMode.SEQUENTIAL,
    )
    device.qachannels[0].readout.run()

    pulse_startQA_string = "QA_GEN_0"
    weight_startQA_string = "QA_INT_0"
    for i in range(number_of_qubits - 1):
        pulse_startQA_string = pulse_startQA_string + f" | QA_GEN_{i+1}"
        weight_startQA_string = weight_startQA_string + f" | QA_INT_{i+1}"

    seqc_program_qa = f"""

    waitDigTrigger(1);              // wait for software trigger to start experiment

    repeat({num_sweep_steps_qubit_spectroscopy}) {{
        repeat({averages_per_sweep_step}) {{
            playZero(4016);
            startQA({pulse_startQA_string}, {weight_startQA_string}, true, 0, 0x0);
        }}

        // Trigger tells control channels to change frequency
        setTrigger(1);
        wait(10);
        setTrigger(0);

    }}
    """

    # configure sequencer
    device.qachannels[0].generator.configure_sequencer_triggering(
        aux_trigger="software_trigger0"
    )
    device.qachannels[0].generator.load_sequencer_program(seqc_program_qa)
    device.qachannels[0].generator.enable_sequencer(single=True)

    print("Sequence of the readout channel:")
    print(seqc_program_qa)


def configure_control_sequences(
    device,
    number_of_qubits,
    min_max_frequencies,
    num_sweep_steps_qubit_spectroscopy,
    qubit_params,
    sgchannel_number,
):
    seqc_program_sg = [[] for _ in range(number_of_qubits)]

    for qubit in range(number_of_qubits):
        seqc_program_sg[
            qubit
        ] = f"""
    const OSC0 = 0;
    const FREQ_START = {min_max_frequencies[qubit][0]};
    const FREQ_STEP = {(min_max_frequencies[qubit][1]-min_max_frequencies[qubit][0])/num_sweep_steps_qubit_spectroscopy};

    configFreqSweep(OSC0, FREQ_START, FREQ_STEP);

    // Frequency sweep
    for(var i = 0; i < {num_sweep_steps_qubit_spectroscopy}; i++) {{
        waitDigTrigger(1);
        setSweepStep(OSC0, i);
    }}
        """
        gain = 0.5 * qubit_params["max_drive_strength"][qubit]
        device.sgchannels[sgchannel_number[qubit]].configure_sine_generation(
            enable=1,
            osc_index=0,
            osc_frequency=qubit_params["qubit_drive_frequency"][qubit],
            gains=[gain, gain, gain, -gain],
        )

        device.sgchannels[sgchannel_number[qubit]].awg.load_sequencer_program(
            seqc_program_sg[qubit]
        )
    print("  ")
    print("Sequence of the control channel for qubit 1:")
    print(seqc_program_sg[0])


def enable_sg_channels(device, sgchannel_number, number_of_qubits, value):

    with device.set_transaction():
        for qubit in range(number_of_qubits):
            device.sgchannels[sgchannel_number[qubit]].output.on(value)


def run_experiment(device, sgchannel_number, number_of_qubits, reenable=False):
    if reenable:
        device.qachannels[0].readout.run()
        device.qachannels[0].generator.enable_sequencer(single=True)

    enable_qa_channel(device, 1)
    enable_sg_channels(device, sgchannel_number, number_of_qubits, 1)

    device.start_continuous_sw_trigger(num_triggers=1, wait_time=2e-3)

    readout_results = device.qachannels[0].readout.read(timeout=200)

    enable_qa_channel(device, 0)
    enable_sg_channels(device, sgchannel_number, number_of_qubits, 0)

    return readout_results


def plot_multiQubit_results_AmpPhase(
    x_axis, readout_results, number_of_qubits, figsize, font_large, font_medium, saveloc
):
    for qubit in range(number_of_qubits):
        fig3, axs = plt.subplots(1, 2, figsize=figsize)
        fig3.suptitle(f"Qubit {qubit}", fontsize=font_large)
        axs[0].plot(x_axis[qubit], np.abs(readout_results[qubit]))
        axs[0].set_title("amplitude [dBm]", fontsize=font_medium)
        axs[0].set_xlabel("qubit drive frequency [MHz]", fontsize=font_medium)
        axs[0].set_ylabel("amplitude [A.U.]", fontsize=font_medium)
        axs[0].tick_params(axis="both", which="major", labelsize=font_medium)

        axs[1].plot(x_axis[qubit], np.unwrap(np.angle(readout_results[qubit])))
        axs[1].set_title("phase [rad]", fontsize=font_medium)
        axs[1].set_xlabel("qubit drive frequency [MHz]", fontsize=font_medium)
        axs[1].set_ylabel("phase [rad]", fontsize=font_medium)
        axs[1].tick_params(axis="both", which="major", labelsize=font_medium)

        plt.savefig(saveloc + f"{qubit}.png")
        plt.show()


def prepare_sequencer_resultLogger_pulsedRabi(
    device,
    number_of_qubits,
    qubit_params,
    wait_factor_in_T1,
    num_averages_rabi_experiment,
    num_steps_rabi_experiment,
    singleShot=False,
):

    pulse_startQA_string = "QA_GEN_0"
    weight_startQA_string = "QA_INT_0"
    for i in range(number_of_qubits - 1):
        pulse_startQA_string = pulse_startQA_string + f" | QA_GEN_{i+1}"
        weight_startQA_string = weight_startQA_string + f" | QA_INT_{i+1}"

    # converts width of single qubit gate assuming its a gaussian to a good waveform length in samples
    # assuming the optimal waveform length is 8 times the width of the gaussian
    single_qubit_pulse_time_seqSamples = int(
        np.max(qubit_params["qubit_single_gate_time"]) * SHFQA_SAMPLING_FREQUENCY
    )
    single_qubit_pulse_time_samples = single_qubit_pulse_time_seqSamples * 8
    readout_pulse_duration_samples = int(
        np.ceil(
            (qubit_params["readout_pulse_duration"] * SHFQA_SAMPLING_FREQUENCY + 10)
            / 16
        )
        * 16
    )
    time_to_next_experiment_seqSamples = int(
        np.max(qubit_params["qubit_T1_time"])
        * wait_factor_in_T1
        * SHFQA_SAMPLING_FREQUENCY
        / 8
    )

    seqc_program_qa = f"""
    waitDigTrigger(1); // wait for software trigger to start experiment
    repeat({num_averages_rabi_experiment}) {{
        repeat({num_steps_rabi_experiment}) {{
            // Trigger control channels to play pulse
            setTrigger(1);
            wait(2);
            setTrigger(0);
            //wait('triggerdelay') // reference point


            // wait until control sequence finished
            wait({single_qubit_pulse_time_seqSamples});

            //start readout
            playZero({readout_pulse_duration_samples});
            startQA({pulse_startQA_string}, {weight_startQA_string}, true, 0, 0x0);

            //wait for qubit decay
            wait({time_to_next_experiment_seqSamples});
        }}
    }}

    """
    device.qachannels[0].generator.load_sequencer_program(seqc_program_qa)

    if singleShot:
        device.qachannels[0].readout.configure_result_logger(
            result_source="integration",
            result_length=num_steps_rabi_experiment * num_averages_rabi_experiment,
            num_averages=1,
            averaging_mode=AveragingMode.CYCLIC,
        )
    else:
        device.qachannels[0].readout.configure_result_logger(
            result_source="integration",
            result_length=num_steps_rabi_experiment,
            num_averages=num_averages_rabi_experiment,
            averaging_mode=AveragingMode.CYCLIC,
        )
    device.qachannels[0].generator.enable_sequencer(single=True)

    print("Sequence of the readout channel:")
    print(seqc_program_qa)

    return (
        single_qubit_pulse_time_samples,
        readout_pulse_duration_samples,
        pulse_startQA_string,
        weight_startQA_string,
    )


def configure_control_sequences_pulsedRabi(
    device,
    number_of_qubits,
    single_qubit_pulse_time_samples,
    qubit_params,
    num_averages_rabi_experiment,
    num_steps_rabi_experiment,
    sgchannel_number,
):

    # Predefine command table
    ct = CommandTable(device.sgchannels[0].awg.commandtable.load_validation_schema())
    ct.table[0].waveform.index = 0
    ct.table[0].amplitude00.value = 0.0
    ct.table[0].amplitude01.value = -0.0
    ct.table[0].amplitude10.value = 0.0
    ct.table[0].amplitude11.value = 0.0
    ct.table[1].waveform.index = 0
    ct.table[1].amplitude00.increment = True
    ct.table[1].amplitude01.increment = True
    ct.table[1].amplitude10.increment = True
    ct.table[1].amplitude11.increment = True

    for qubit in range(number_of_qubits):
        with device.set_transaction():
            device.sgchannels[sgchannel_number[qubit]].sines[0].i.enable(0)
            device.sgchannels[sgchannel_number[qubit]].sines[0].q.enable(0)
            device.sgchannels[sgchannel_number[qubit]].awg.modulation.enable(1)
            device.sgchannels[sgchannel_number[qubit]].oscs[0].freq(
                qubit_params["qubit_drive_frequency"][qubit]
            )

        # Upload waveform
        seqc = inspect.cleandoc(
            f"""
        // Define a single waveform
        wave rabi_pulse=gauss({single_qubit_pulse_time_samples}, 1, {single_qubit_pulse_time_samples/2}, {qubit_params['qubit_single_gate_time'][qubit]*SHFQA_SAMPLING_FREQUENCY});

        // Assign a dual channel waveform to wave table entry
        assignWaveIndex(1,2,rabi_pulse, 1,2,rabi_pulse, 0);
        resetOscPhase();
        repeat ({num_averages_rabi_experiment}) {{
            waitDigTrigger(1);
            executeTableEntry(0);
            repeat ({num_steps_rabi_experiment}-1) {{
                waitDigTrigger(1);
                executeTableEntry(1);
            }}
        }}
        """
        )

        device.sgchannels[sgchannel_number[qubit]].awg.load_sequencer_program(seqc)

        # update command table
        increment_value = qubit_params["max_drive_strength"][qubit] / (
            num_steps_rabi_experiment - 1
        )
        ct.table[1].amplitude00.value = increment_value
        ct.table[1].amplitude01.value = -increment_value
        ct.table[1].amplitude10.value = increment_value
        ct.table[1].amplitude11.value = increment_value

        device.sgchannels[sgchannel_number[qubit]].awg.commandtable.upload_to_device(ct)

    print("  ")
    print("Sequence of the control channel for last qubit:")
    print(seqc)

    print("  ")
    print("command table of the control channel for last qubit:")
    print(ct)


def configure_control_sequences_pulsedRamsey(
    device,
    number_of_qubits,
    single_qubit_pulse_time_samples,
    qubit_params,
    num_averages,
    num_steps,
    wait_time_steps_samples,
    sgchannel_number,
):

    # Predefine command table
    ct = CommandTable(device.sgchannels[0].awg.commandtable.load_validation_schema())
    ct.table[0].waveform.index = 0
    ct.table[0].amplitude00.value = 0.0
    ct.table[0].amplitude01.value = -0.0
    ct.table[0].amplitude10.value = 0.0
    ct.table[0].amplitude11.value = 0.0

    for qubit in range(number_of_qubits):
        seqc = inspect.cleandoc(
            f"""
            // Define a single waveform
            wave rabi_pulse=gauss({single_qubit_pulse_time_samples}, 1, {single_qubit_pulse_time_samples/2}, {qubit_params['qubit_single_gate_time'][qubit]*SHFQA_SAMPLING_FREQUENCY});

            // Assign a dual channel waveform to wave table entry
            assignWaveIndex(1,2,rabi_pulse, 1,2,rabi_pulse, 0);
            resetOscPhase();
            var time_ind = 0;
            repeat ({num_averages}) {{
                for (time_ind = 0; time_ind < {num_steps}; time_ind++) {{
                    waitDigTrigger(1);
                    executeTableEntry(0);
                    playZero({wait_time_steps_samples}*time_ind);
                    executeTableEntry(0);
                }}
            }}
        """
        )
        device.sgchannels[sgchannel_number[qubit]].awg.load_sequencer_program(seqc)

        ct.table[0].amplitude00.value = qubit_params["qubit_pi/2_amplitude"][qubit]
        ct.table[0].amplitude01.value = -qubit_params["qubit_pi/2_amplitude"][qubit]
        ct.table[0].amplitude10.value = qubit_params["qubit_pi/2_amplitude"][qubit]
        ct.table[0].amplitude11.value = qubit_params["qubit_pi/2_amplitude"][qubit]

        device.sgchannels[sgchannel_number[qubit]].awg.commandtable.upload_to_device(ct)

        print("  ")
        print("Sequence of the control channel for last qubit:")
        print(seqc)

        print("  ")
        print("command table of the control channel for last qubit:")
        print(ct)


def configure_control_sequences_pulsedT1(
    device,
    number_of_qubits,
    single_qubit_pulse_time_samples,
    qubit_params,
    num_averages,
    num_steps,
    sgchannel_number,
):

    # Predefine command table
    ct = CommandTable(device.sgchannels[0].awg.commandtable.load_validation_schema())
    ct.table[0].waveform.index = 0
    ct.table[0].amplitude00.value = 0.0
    ct.table[0].amplitude01.value = -0.0
    ct.table[0].amplitude10.value = 0.0
    ct.table[0].amplitude11.value = 0.0

    for qubit in range(number_of_qubits):
        seqc = inspect.cleandoc(
            f"""
            // Define a single waveform
            wave rabi_pulse=gauss({single_qubit_pulse_time_samples}, 1, {single_qubit_pulse_time_samples/2}, {qubit_params['qubit_single_gate_time'][qubit]*SHFQA_SAMPLING_FREQUENCY});

            // Assign a dual channel waveform to wave table entry
            assignWaveIndex(1,2,rabi_pulse, 1,2,rabi_pulse, 0);
            resetOscPhase();
            var time_ind = 0;
            repeat ({num_averages}) {{
                for (time_ind = 0; time_ind < {num_steps}; time_ind++) {{
                    waitDigTrigger(1);
                    executeTableEntry(0);
                }}
            }}
            """
        )

        device.sgchannels[sgchannel_number[qubit]].awg.load_sequencer_program(seqc)

        ct.table[0].amplitude00.value = qubit_params["qubit_pi_amplitude"][qubit]
        ct.table[0].amplitude01.value = -qubit_params["qubit_pi_amplitude"][qubit]
        ct.table[0].amplitude10.value = qubit_params["qubit_pi_amplitude"][qubit]
        ct.table[0].amplitude11.value = qubit_params["qubit_pi_amplitude"][qubit]

        device.sgchannels[sgchannel_number[qubit]].awg.commandtable.upload_to_device(ct)

        print("  ")
        print("Sequence of the control channel for last qubit:")
        print(seqc)

        print("  ")
        print("command table of the control channel for last qubit:")
        print(ct)


def configure_control_sequences_pulsedRB(
    device,
    number_of_qubits,
    clifford_waves,
    clifford_len,
    qubit_params,
    num_averages,
    sgchannel_number,
):
    wait_factor_in_T1 = 0

    # Waveform definition - allocating the waveofmr memory for the seqC program
    waveforms_def = ""
    for i, wave in enumerate(clifford_waves):
        wave_len = len(wave)
        waveforms_def += f"assignWaveIndex(1,2,placeholder({wave_len:d}),1,2, placeholder({wave_len:d}), {i:d});\n"

    # define user registers that will be used in the seqC
    AWG_REGISTER_M1 = 0
    AWG_REGISTER_SEED = 1
    AWG_REGISTER_RECOVERY = 2

    # converts width of single qubit gate assuming its a gaussian to a good waveform length in samples
    # assuming the optimal waveform length is 8 times the width of the gaussian
    time_to_next_experiment_seqSamples = int(
        np.max(qubit_params["qubit_T1_time"])
        * wait_factor_in_T1
        * SHFQA_SAMPLING_FREQUENCY
        / 8
    )

    # Predefine command table
    ct = CommandTable(device.sgchannels[0].awg.commandtable.load_validation_schema())
    for i in range(clifford_len):
        ct.table[i].waveform.index = 0
        ct.table[i].amplitude00.value = 1
        ct.table[i].amplitude01.value = -1
        ct.table[i].amplitude10.value = 1
        ct.table[i].amplitude11.value = 1

    waveforms = Waveforms()
    for i, wave in enumerate(clifford_waves):
        waveforms[i] = wave

    for qubit in range(number_of_qubits):
        seqc = inspect.cleandoc(
            f"""
            const CLIFFORD_LEN = 24;

            //Waveform definition
            {waveforms_def:s}

            // Runtime parameters, set by user registers:
            // sequence length
            var m1 = getUserReg({AWG_REGISTER_M1:d});
            // PRNG seed
            var seed = getUserReg({AWG_REGISTER_SEED:d});
            // recovery gate index
            var recovery = getUserReg({AWG_REGISTER_RECOVERY:d});

            //Configure the PRNG
            setPRNGSeed(seed);
            setPRNGRange(0, CLIFFORD_LEN);

            // trigger at start of each sequence
            waitDigTrigger(1);

            repeat ({num_averages}) {{
                //Reset the oscillator phase
                resetOscPhase();
                wait(5);

                // (Pseudo) Random sequence of command table entries
                repeat (m1) {{
                    var gate = getPRNGValue();
                    executeTableEntry(gate);
                }}
                // Final recovery gate
                executeTableEntry(recovery);

                wait(4);

                setTrigger(1);
                wait(5);
                setTrigger(0);

                //wait for qubit decay
                wait({time_to_next_experiment_seqSamples});
            }}
            """
        )

        device.sgchannels[sgchannel_number[qubit]].awg.load_sequencer_program(seqc)
        device.sgchannels[sgchannel_number[qubit]].awg.commandtable.upload_to_device(ct)

        # upload waveforms to instrument
        device.sgchannels[sgchannel_number[qubit]].awg.write_to_waveform_memory(
            waveforms
        )
        print("waveforms uploaded")


def plot_multiQubit_results_IQ(
    x_axis,
    readout_results,
    number_of_qubits,
    figsize,
    font_large,
    font_medium,
    label="pulse amplitude [A.U.]",
    saveloc="",
):
    for qubit in range(number_of_qubits):
        fig3, axs = plt.subplots(1, 2, figsize=figsize)
        fig3.suptitle(f"Qubit {qubit}", fontsize=font_large)
        axs[0].plot(x_axis, np.real(readout_results[qubit] - readout_results[qubit][0]))
        axs[0].set_title("I quadrature [A.U.]", fontsize=font_medium)
        axs[0].set_xlabel(label, fontsize=font_medium)
        axs[0].set_ylabel("quadrature [A.U.]", fontsize=font_medium)
        axs[0].tick_params(axis="both", which="major", labelsize=font_medium)

        axs[1].plot(x_axis, np.imag(readout_results[qubit] - readout_results[qubit][0]))
        axs[1].set_title("Q quadrature [A.U.]", fontsize=font_medium)
        axs[1].set_xlabel(label, fontsize=font_medium)
        axs[1].set_ylabel("quadrature [A.U.]", fontsize=font_medium)
        axs[1].tick_params(axis="both", which="major", labelsize=font_medium)

        plt.savefig(saveloc + f"{qubit}.png")
        plt.show()


def plot_multiQubit_results_singleShot(
    readout_results, number_of_qubits, figsize, font_large, saveloc
):
    for qubit in range(number_of_qubits):
        fig3 = plt.figure(figsize=figsize)
        fig3.suptitle(f"Qubit {qubit}", fontsize=font_large)
        plt.plot(
            np.real(readout_results[qubit] - readout_results[qubit][0]),
            np.imag(readout_results[qubit] - readout_results[qubit][0]),
            ".",
        )
        plt.xlabel("integrated I Quadrature [A.U.]", fontsize=font_large)
        plt.ylabel("integrated Q Quadrature [A.U.]", fontsize=font_large)
        plt.tick_params(axis="both", which="major", labelsize=font_large)

    plt.savefig(saveloc)
    plt.show()


# fitting function for amplitude Rabi
def amplitude_rabi(amp, Omega, amplitude):
    return -amplitude * np.cos(amp * Omega / 2) + amplitude


# fitting function for amplitude Rabi
def ramsey_time_exp(time, detuning, amplitude, T2, phase):
    return (
        -amplitude * np.cos(2 * np.pi * detuning * time + phase) * np.exp(-time / T2)
        + amplitude
    )


def ramsey_time_gauss(time, detuning, amplitude, T2, phase):
    return (
        -amplitude
        * np.cos(2 * np.pi * detuning * time + phase)
        * np.exp(-(time ** 2) / 2 / T2 ** 2)
        + amplitude
    )


# fitting function for amplitude Rabi
def T1_time_exp(time, amplitude, T1):
    return amplitude * np.exp(-time / T1)


# TODO: pulse envelepo: complex drag pulse, add qscale


def pulse_envelope(
    amplitude, length, phase, qscale=-1.0, sigma=1 / 3, sample_rate=2e9, tol=15
):
    # sigma = 1/3
    # ensure waveform length is integer multiple of 16
    samples = round(sample_rate * length / 16) * 16
    x = np.linspace(-1, 1, samples)
    # output is complex, where phase later determines the gate rotation axis
    ybare = (
        np.exp(-(x ** 2) / 2 / sigma ** 2)
        * (1 - 1j * qscale * x / sigma ** 2 * sigma ** 2)
        * np.exp(1j * np.deg2rad(phase))
    )

    # ybare = np.exp(-x**2 / sigma**2)*(1)*np.exp(1j * np.deg2rad(phase))
    # norm = np.norm(ybare)
    # y = amplitude * ybare/norm
    # y=amplitude*ybare/np.max(np.abs(ybare))*1/np.sqrt(2*np.pi)/sigma
    y = amplitude * ybare
    return y.round(tol)
