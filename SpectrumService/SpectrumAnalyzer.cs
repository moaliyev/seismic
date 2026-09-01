namespace SpectrumService;

/// <summary>Average amplitude spectrum of a block of seismic traces.</summary>
internal static class SpectrumAnalyzer
{
    public const string HannWindow = "hann";
    public const string NoWindow = "none";

    /// <summary>
    /// Averages the one-sided amplitude spectrum over every trace in the block.
    ///
    /// <paramref name="samples"/> is row-major: one trace per row, time running
    /// along the row. Traces are zero-padded to the next power of two, so the
    /// bin spacing is 1 / (nfft * sampleInterval).
    /// </summary>
    public static SpectrumResult Analyze(
        float[] samples, int rows, int cols, double sampleInterval, string window, bool detrend)
    {
        if (rows <= 0 || cols <= 0)
        {
            throw new ArgumentException($"empty region: {rows}x{cols}");
        }

        if (cols < 2)
        {
            throw new ArgumentException("a spectrum needs at least two samples per trace");
        }

        if (sampleInterval <= 0)
        {
            throw new ArgumentException($"sample interval must be positive, got {sampleInterval}");
        }

        var nfft = Fft.NextPowerOfTwo(cols);
        var fft = new Fft(nfft);
        var taper = BuildWindow(window, cols);

        var gain = 0.0;
        foreach (var weight in taper)
        {
            gain += weight;
        }

        var bins = (nfft / 2) + 1;
        var accumulated = new double[bins];
        var re = new double[nfft];
        var im = new double[nfft];

        for (var row = 0; row < rows; row++)
        {
            var offset = row * cols;

            // Removing the mean keeps a DC offset from swamping the low end
            var mean = 0.0;
            if (detrend)
            {
                for (var i = 0; i < cols; i++)
                {
                    mean += samples[offset + i];
                }

                mean /= cols;
            }

            for (var i = 0; i < cols; i++)
            {
                re[i] = (samples[offset + i] - mean) * taper[i];
            }

            Array.Clear(re, cols, nfft - cols);
            Array.Clear(im, 0, nfft);

            fft.Forward(re, im);

            for (var k = 0; k < bins; k++)
            {
                accumulated[k] += Math.Sqrt((re[k] * re[k]) + (im[k] * im[k]));
            }
        }

        // Scale so a pure sine reads back at its own amplitude: divide out the
        // trace count and the window's coherent gain, then fold the negative
        // frequencies onto the positive ones (DC and Nyquist have no twin).
        var amplitudes = new float[bins];
        var scale = 1.0 / (rows * gain);
        for (var k = 0; k < bins; k++)
        {
            var oneSided = (k == 0 || k == bins - 1) ? scale : 2.0 * scale;
            amplitudes[k] = (float)(accumulated[k] * oneSided);
        }

        return new SpectrumResult(amplitudes, nfft, 1.0 / (nfft * sampleInterval), rows);
    }

    private static double[] BuildWindow(string window, int length)
    {
        var taper = new double[length];
        if (string.Equals(window, NoWindow, StringComparison.OrdinalIgnoreCase))
        {
            Array.Fill(taper, 1.0);
            return taper;
        }

        if (!string.Equals(window, HannWindow, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"unknown window '{window}', expected 'hann' or 'none'");
        }

        for (var i = 0; i < length; i++)
        {
            taper[i] = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (length - 1)));
        }

        return taper;
    }
}

/// <summary>Spectrum plus the numbers the caller needs to build a frequency axis.</summary>
internal readonly record struct SpectrumResult(float[] Amplitudes, int Nfft, double BinWidth, int Traces);
