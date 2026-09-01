using System.Numerics;

namespace SpectrumService;

/// <summary>
/// In-place iterative radix-2 Cooley-Tukey FFT.
///
/// Written out by hand rather than pulled from NuGet so the module builds with
/// nothing but the .NET SDK. One instance is tied to one transform length and
/// caches its twiddle factors, so analysing thousands of traces of the same
/// length only pays for the trigonometry once.
/// </summary>
internal sealed class Fft
{
    private readonly int _size;
    private readonly double[] _cos;
    private readonly double[] _sin;
    private readonly int[] _reversed;

    public Fft(int size)
    {
        if (size < 2 || (size & (size - 1)) != 0)
        {
            throw new ArgumentException($"transform length must be a power of two, got {size}", nameof(size));
        }

        _size = size;
        _cos = new double[size / 2];
        _sin = new double[size / 2];
        for (var i = 0; i < size / 2; i++)
        {
            var angle = -2.0 * Math.PI * i / size;
            _cos[i] = Math.Cos(angle);
            _sin[i] = Math.Sin(angle);
        }

        var bits = BitOperations.Log2((uint)size);
        _reversed = new int[size];
        for (var i = 0; i < size; i++)
        {
            var reversed = 0;
            for (var bit = 0; bit < bits; bit++)
            {
                if ((i & (1 << bit)) != 0)
                {
                    reversed |= 1 << (bits - 1 - bit);
                }
            }

            _reversed[i] = reversed;
        }
    }

    public int Size => _size;

    /// <summary>Rounds up to the next power of two, which is the length we zero-pad to.</summary>
    public static int NextPowerOfTwo(int n)
    {
        var size = 1;
        while (size < n)
        {
            size <<= 1;
        }

        return size;
    }

    /// <summary>Transforms <paramref name="re"/>/<paramref name="im"/> in place.</summary>
    public void Forward(double[] re, double[] im)
    {
        if (re.Length != _size || im.Length != _size)
        {
            throw new ArgumentException($"expected buffers of length {_size}");
        }

        for (var i = 0; i < _size; i++)
        {
            var j = _reversed[i];
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        for (var length = 2; length <= _size; length <<= 1)
        {
            var half = length >> 1;
            var stride = _size / length;
            for (var start = 0; start < _size; start += length)
            {
                for (var k = 0; k < half; k++)
                {
                    var twiddle = k * stride;
                    var wRe = _cos[twiddle];
                    var wIm = _sin[twiddle];

                    var top = start + k;
                    var bottom = top + half;

                    var oddRe = (re[bottom] * wRe) - (im[bottom] * wIm);
                    var oddIm = (re[bottom] * wIm) + (im[bottom] * wRe);

                    re[bottom] = re[top] - oddRe;
                    im[bottom] = im[top] - oddIm;
                    re[top] += oddRe;
                    im[top] += oddIm;
                }
            }
        }
    }
}
