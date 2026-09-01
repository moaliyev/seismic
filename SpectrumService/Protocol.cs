using System.Buffers.Binary;
using System.Text;

namespace SpectrumService;

/// <summary>
/// Framing for the pipe the host application talks over.
///
///   request   "SPQ1" | uint32 headerLength | header JSON (UTF-8) | float32[rows * cols]
///   response  "SPR1" | uint32 headerLength | header JSON (UTF-8) | float32[bins]
///
/// Integers are little-endian and the sample block is row-major, one trace per
/// row. Shape and options travel in the JSON header so the binary block can stay
/// a flat run of floats that both sides copy without parsing.
/// </summary>
internal static class Protocol
{
    public const string RequestMagic = "SPQ1";
    public const string ResponseMagic = "SPR1";

    private const int MaxHeaderBytes = 64 * 1024;
    private const long MaxPayloadFloats = 128L * 1024 * 1024;

    /// <summary>Reads exactly <paramref name="count"/> bytes, or null if the stream is already at its end.</summary>
    public static byte[]? ReadExact(Stream stream, int count)
    {
        var buffer = new byte[count];
        var filled = 0;
        while (filled < count)
        {
            var read = stream.Read(buffer, filled, count - filled);
            if (read == 0)
            {
                if (filled == 0)
                {
                    return null;
                }

                throw new EndOfStreamException($"stream ended after {filled} of {count} bytes");
            }

            filled += read;
        }

        return buffer;
    }

    /// <summary>Reads the next frame header, or null once the host closes the pipe.</summary>
    public static string? ReadHeader(Stream stream, string expectedMagic)
    {
        var magic = ReadExact(stream, 4);
        if (magic is null)
        {
            return null;
        }

        var actual = Encoding.ASCII.GetString(magic);
        if (actual != expectedMagic)
        {
            throw new InvalidDataException($"expected magic '{expectedMagic}', got '{actual}'");
        }

        var lengthBytes = ReadExact(stream, 4)
            ?? throw new EndOfStreamException("frame ended before the header length");
        var length = BinaryPrimitives.ReadUInt32LittleEndian(lengthBytes);
        if (length > MaxHeaderBytes)
        {
            throw new InvalidDataException($"header of {length} bytes exceeds the {MaxHeaderBytes} byte limit");
        }

        var header = ReadExact(stream, (int)length)
            ?? throw new EndOfStreamException("frame ended before the header");
        return Encoding.UTF8.GetString(header);
    }

    public static float[] ReadFloats(Stream stream, long count)
    {
        if (count < 0 || count > MaxPayloadFloats)
        {
            throw new InvalidDataException($"payload of {count} samples is out of range");
        }

        var bytes = ReadExact(stream, checked((int)(count * sizeof(float))))
            ?? throw new EndOfStreamException("frame ended before the payload");
        var values = new float[count];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    public static void WriteFrame(Stream stream, string magic, string header, float[] payload)
    {
        var headerBytes = Encoding.UTF8.GetBytes(header);
        var lengthBytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(lengthBytes, (uint)headerBytes.Length);

        var payloadBytes = new byte[payload.Length * sizeof(float)];
        Buffer.BlockCopy(payload, 0, payloadBytes, 0, payloadBytes.Length);

        stream.Write(Encoding.ASCII.GetBytes(magic));
        stream.Write(lengthBytes);
        stream.Write(headerBytes);
        stream.Write(payloadBytes);
        stream.Flush();
    }
}
