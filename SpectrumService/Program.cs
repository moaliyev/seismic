using System.Text.Json;
using System.Text.Json.Serialization;

namespace SpectrumService;

/// <summary>
/// Standalone spectrum module for the seismic viewer.
///
/// It is a plain console executable that owns no UI and no data: the host
/// application starts it once, then streams request frames on stdin and reads
/// response frames from stdout for as long as it needs spectra. Diagnostics go
/// to stderr, which keeps the stdout pipe purely binary. See Protocol.cs for
/// the frame layout.
/// </summary>
internal static class Program
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        // camelCase on the wire, but tolerant when reading, so the header keys
        // match the documented format exactly
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    private static int Main(string[] args)
    {
        if (!BitConverter.IsLittleEndian)
        {
            Console.Error.WriteLine("SpectrumService requires a little-endian host");
            return 2;
        }

        if (args.Length > 0 && args[0] == "--selftest")
        {
            return SelfTest();
        }

        using var input = Console.OpenStandardInput();
        using var output = Console.OpenStandardOutput();

        while (true)
        {
            string? header;
            try
            {
                header = Protocol.ReadHeader(input, Protocol.RequestMagic);
            }
            catch (Exception exception)
            {
                // The stream is out of sync, so there is no safe place to resume
                Console.Error.WriteLine($"framing error: {exception.Message}");
                return 1;
            }

            if (header is null)
            {
                return 0; // host closed the pipe: a normal shutdown
            }

            try
            {
                Respond(output, header, input);
            }
            catch (EndOfStreamException exception)
            {
                Console.Error.WriteLine($"truncated request: {exception.Message}");
                return 1;
            }
        }
    }

    private static void Respond(Stream output, string header, Stream input)
    {
        SpectrumRequest request;
        try
        {
            request = JsonSerializer.Deserialize<SpectrumRequest>(header, JsonOptions)
                      ?? throw new InvalidDataException("empty request header");
        }
        catch (JsonException exception)
        {
            WriteError(output, $"malformed request header: {exception.Message}");
            return;
        }

        if (request.Rows <= 0 || request.Cols <= 0)
        {
            WriteError(output, $"invalid region shape {request.Rows}x{request.Cols}");
            return;
        }

        // The payload has to be drained even for a request we reject, otherwise
        // the next read would start mid-block and desynchronise the pipe.
        var samples = Protocol.ReadFloats(input, (long)request.Rows * request.Cols);

        try
        {
            var result = SpectrumAnalyzer.Analyze(
                samples, request.Rows, request.Cols, request.SampleInterval, request.Window, request.Detrend);

            var responseHeader = JsonSerializer.Serialize(
                new SpectrumResponse
                {
                    Status = "ok",
                    Bins = result.Amplitudes.Length,
                    BinWidth = result.BinWidth,
                    Nfft = result.Nfft,
                    Traces = result.Traces,
                },
                JsonOptions);

            Protocol.WriteFrame(output, Protocol.ResponseMagic, responseHeader, result.Amplitudes);
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidDataException)
        {
            WriteError(output, exception.Message);
        }
    }

    private static void WriteError(Stream output, string message)
    {
        var header = JsonSerializer.Serialize(
            new SpectrumResponse { Status = "error", Message = message }, JsonOptions);
        Protocol.WriteFrame(output, Protocol.ResponseMagic, header, Array.Empty<float>());
    }

    /// <summary>
    /// Checks the transform against a signal whose answer is known: a 25 Hz sine
    /// sampled at 4 ms must peak in the 25 Hz bin at its own amplitude.
    /// </summary>
    private static int SelfTest()
    {
        const int rows = 8;
        const int cols = 512;
        const double sampleInterval = 0.004;
        const double frequency = 25.0;
        const double amplitude = 3.0;

        var samples = new float[rows * cols];
        for (var row = 0; row < rows; row++)
        {
            for (var i = 0; i < cols; i++)
            {
                samples[(row * cols) + i] =
                    (float)(amplitude * Math.Sin(2.0 * Math.PI * frequency * i * sampleInterval));
            }
        }

        var result = SpectrumAnalyzer.Analyze(samples, rows, cols, sampleInterval, "hann", true);

        var peak = 0;
        for (var k = 1; k < result.Amplitudes.Length; k++)
        {
            if (result.Amplitudes[k] > result.Amplitudes[peak])
            {
                peak = k;
            }
        }

        var peakFrequency = peak * result.BinWidth;
        var peakAmplitude = result.Amplitudes[peak];
        Console.Error.WriteLine(
            $"bins={result.Amplitudes.Length} nfft={result.Nfft} df={result.BinWidth:F4} Hz");
        Console.Error.WriteLine(
            $"peak at {peakFrequency:F2} Hz (expected {frequency:F2}), amplitude {peakAmplitude:F3} (expected {amplitude:F3})");

        var ok = Math.Abs(peakFrequency - frequency) <= result.BinWidth
                 && Math.Abs(peakAmplitude - amplitude) < 0.15 * amplitude;
        Console.Error.WriteLine(ok ? "selftest ok" : "selftest FAILED");
        return ok ? 0 : 1;
    }
}

internal sealed record SpectrumRequest
{
    public int Rows { get; init; }

    public int Cols { get; init; }

    /// <summary>Time between samples, in seconds.</summary>
    public double SampleInterval { get; init; } = 0.004;

    public string Window { get; init; } = SpectrumAnalyzer.HannWindow;

    public bool Detrend { get; init; } = true;
}

internal sealed record SpectrumResponse
{
    public string Status { get; init; } = "ok";

    public string? Message { get; init; }

    public int Bins { get; init; }

    /// <summary>Frequency step between neighbouring bins, in Hz.</summary>
    public double BinWidth { get; init; }

    public int Nfft { get; init; }

    public int Traces { get; init; }
}
