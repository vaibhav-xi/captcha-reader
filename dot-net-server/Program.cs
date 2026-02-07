using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;
using System.IO;
using System.Linq;
using System.Collections.Generic;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace dot_net_server;

public class Program
{
    public static void Main(string[] args)
    {
        NativeLibrary.SetDllImportResolver(
            typeof(InferenceSession).Assembly,
            ResolveOnnx);

        var builder = WebApplication.CreateBuilder(args);

        builder.WebHost.ConfigureKestrel(o =>
        {
            o.Limits.MaxConcurrentConnections = 1000;
            o.Limits.MaxRequestBodySize = 5 * 1024 * 1024;
        });

        var app = builder.Build();

        var session = new InferenceSession("captcha_ctc.onnx");

        string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@=#";

        string saveDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "captchas");

        Directory.CreateDirectory(saveDir);

        app.MapGet("/health", () => Results.Ok(new
        {
            ok = true,
            model = "onnx",
            time = DateTime.UtcNow
        }));

        app.MapPost("/api/captcha", async (CaptchaRequest req) =>
        {
            var sw = Stopwatch.StartNew();

            if (string.IsNullOrWhiteSpace(req.Base64))
                return Results.BadRequest("base64 missing");

            var b64 = req.Base64;
            int comma = b64.IndexOf(',');
            if (comma > 0)
                b64 = b64[(comma + 1)..];

            byte[] bytes = Convert.FromBase64String(b64);

            using var img = Image.Load<Rgba32>(bytes);
            img.Mutate(x => x.Resize(200, 50).Grayscale());

            float[] data = new float[50 * 200];
            int idx = 0;

            for (int y = 0; y < 50; y++)
            for (int x = 0; x < 200; x++)
                data[idx++] = img[x, y].R / 255f;

            var tensor = new DenseTensor<float>(data, new[] { 1, 50, 200, 1 });

            using var results = session.Run(
                new[] { NamedOnnxValue.CreateFromTensor("image", tensor) });

            var output = results.First().AsTensor<float>();

            var text = DecodeCTC(
                output,
                output.Dimensions[1],
                output.Dimensions[2],
                chars);

            sw.Stop();

            return Results.Ok(new { ok = true, text, ms = sw.ElapsedMilliseconds });
        });

        app.Run("http://127.0.0.1:5077");
    }

    private static IntPtr ResolveOnnx(
        string libraryName,
        Assembly assembly,
        DllImportSearchPath? path)
    {
        if (libraryName == "onnxruntime")
        {
            var full = Path.Combine(
                AppContext.BaseDirectory,
                "runtimes",
                "osx-arm64",
                "native",
                "libonnxruntime.dylib");

            Console.WriteLine("Resolver loading: " + full);

            if (File.Exists(full))
                return NativeLibrary.Load(full);
        }

        return IntPtr.Zero;
    }

    static string DecodeCTC(Tensor<float> t, int time, int classes, string chars)
    {
        List<int> seq = new();

        for (int i = 0; i < time; i++)
        {
            int best = 0;
            float bestVal = float.MinValue;

            for (int c = 0; c < classes; c++)
            {
                float v = t[0, i, c];
                if (v > bestVal)
                {
                    bestVal = v;
                    best = c;
                }
            }

            seq.Add(best);
        }

        int blank = classes - 1;
        List<int> cleaned = new();
        int prev = -1;

        foreach (var s in seq)
        {
            if (s != prev && s != blank)
                cleaned.Add(s);
            prev = s;
        }

        return new string(cleaned.Select(i => chars[i]).ToArray());
    }
}

public record CaptchaRequest(
    [property: JsonPropertyName("base64")] string Base64,
    [property: JsonPropertyName("fileName")] string? FileName);
