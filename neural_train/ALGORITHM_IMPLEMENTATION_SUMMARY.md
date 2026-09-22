# Echo Cancellation Algorithm Implementation Summary

## Current Status

The RLS Bisheng MDF algorithm has been successfully integrated into `test_subband_echo_cancellation.py`. Both LMS and RLS algorithms are now available for testing.

## How to Run

```bash
# Run with RLS Bisheng MDF (frequency-domain, more complex) - DEFAULT
python test_subband_echo_cancellation.py rls

# Run with NLMS (time-domain, simpler)
python test_subband_echo_cancellation.py lms
```

## Algorithm Differences

### NLMS (Normalized Least Mean Squares)
- **Domain**: Time-domain
- **Complexity**: Low - simple gradient descent
- **Update Rule**: `w = w + mu * e * x / (x^T * x + eps)`
- **Processing**: Sample-by-sample within blocks
- **Convergence**: Slower, but stable
- **Parameters**: `mu` (step size), typically 0.1

### RLS Bisheng MDF (Recursive Least Squares Multi-Delay Filter)
- **Domain**: Frequency-domain  
- **Complexity**: High - uses Toeplitz matrix computations
- **Update Rule**: RLS with search direction and adaptive step size
- **Processing**: Block-based with FFT/IFFT
- **Convergence**: Faster, more computationally intensive
- **Parameters**: `alpha` (learning rate), `beta` (forgetting factor)
- **Reference**: Based on Valin (2007) and Sayed (2003)

## Implementation Architecture

The `BlockRLSBishengMDF` class wraps the frequency-domain RLS algorithm:

1. **Time-to-Frequency Conversion**: 
   - Applies Hanning window to reduce edge effects
   - Performs FFT to convert to frequency domain
   - Reshapes to (nbin, 1) for RLS filter input

2. **RLS Processing**:
   - Uses `RLSBishengMDF` from `rls_bisheng_mdf.py`
   - Maintains buffer history for proper block processing
   - Applies RLS update with Toeplitz matrix structure

3. **Frequency-to-Time Conversion**:
   - Performs IFFT to convert back to time domain
   - Extracts valid samples using overlap-save method

## Echo Estimation Visualization

The test now generates graphs showing:
1. Original speech signal
2. Speech with echo (microphone signal)
3. Processed signal (after echo cancellation)
4. **Estimated echo** (what the algorithm removed)

Files generated:
- `echo_cancellation_waveforms_{algorithm}.png` - Time domain visualization
- `echo_cancellation_spectrograms_{algorithm}.png` - Frequency domain visualization

## Verification Metrics

The implementation includes algorithm verification that checks:
- Echo estimation correlation (how well estimated echo matches true echo)
- Echo estimation accuracy in dB
- Filter weight analysis (for NLMS: delay and decay accuracy)
- For RLS: frequency domain weight statistics

## Test Scenario

The test uses a **single-talk scenario**:
```
mic_signal = original_speech + echo_of_original_speech
echo = 0.4 * delayed(original_speech, 50ms)
```

**Goal**: Remove echo component from mic_signal to recover original_speech

**Challenge**: Since both mic and reference contain the same speech content, 
the adaptive filter must learn to identify and remove only the delayed+attenuated 
component (the echo), not the entire signal.

## Output Files

After running the test:
- `processed_signal_{algorithm}.wav` - Echo-cancelled audio
- `echo_cancellation_waveforms_{algorithm}.png` - Waveform comparison
- `echo_cancellation_spectrograms_{algorithm}.png` - Spectrogram comparison
- `echo_cancellation_validation.png` - Detailed validation plot (from validation script)

## Key Implementation Details

### Block Processing with Overlap-Save (75% overlap)
- Block size: 1614 samples (73.2 ms at 22050 Hz)
- Step size: 403 samples (25% of block)
- Overlap: 1211 samples (75% of block)
- Only the last 25% of each block's output is "new" data

### Filter Configuration
- Filter length: echo_delay + 512 samples
- Covers approximately 73 ms of echo path
- RLS uses N_G=1 (single block filter)

## Running Validation

To get detailed performance metrics:
```bash
python validate_echo_cancellation.py
```

This compares the processed signals against the original and provides:
- ERLE (Echo Return Loss Enhancement)
- Echo estimation correlation
- RMSE measurements
- SNR improvement

## Notes on Performance

The single-talk test scenario is particularly challenging because:
1. Reference and desired signals are highly correlated
2. The filter must distinguish between original and echo components
3. Both components come from the same source signal

In real-world scenarios with actual far-end and near-end speakers (uncorrelated signals), 
the echo cancellation would perform much better.

## Future Improvements

1. **Better Test Scenario**: Use uncorrelated far-end and near-end signals
2. **Double-talk Detection**: Add logic to handle when both parties speak
3. **Multi-configuration RLS**: Use `RLSBishengMDF_Multi` for better performance
4. **Parameter Tuning**: Optimize alpha, beta, and other RLS parameters
5. **Nonlinear Processing**: Add residual echo suppression
