# Lab LLM ASR Translation

...Description...

## 🎧 Pre-Processing: Improve Audio Quality

The first step focuses on improving the audio quality:

1. **Noise Reduction**  
   Suppresses background noise such as hiss, hum, or ambient sounds to enhance speech clarity.

2. **Bandpass Filtering**  
   Applies high-pass and low-pass filters to isolate the core speech frequency range.

3. **Dynamic Range Compression**  
   Reduces differences between quiet and loud segments to maintain consistent intelligibility.

4. **Loudness Normalization**  
   Normalizes overall loudness to a standard level, suitable for ASR engines.


5. **Format Standardization**  
   Converts audio to the optimal format:
   - Mono Channel
   - 16-bit PCM
   - 16,000 Hz Sample Rate
