using System;
using System.Diagnostics;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;
using System.Net.Http.Json;

namespace dot_net_server;

public class Program
{
    static HttpClient http = new();
    static string pythonApi = "http://localhost:8000/predict";

    public static void Main(string[] args)
    {
        Console.WriteLine("=== BOOT ===");

        var builder = WebApplication.CreateBuilder(args);
        var app = builder.Build();

        app.MapPost("/api/captcha", async (CaptchaRequest req) =>
        {
            Console.WriteLine("\n================ REQUEST ================");
            var sw = Stopwatch.StartNew();

            var b64 = req.Base64;
            int comma = b64.IndexOf(',');
            if (comma > 0)
                b64 = b64[(comma + 1)..];

            byte[] bytes = Convert.FromBase64String(b64);

            Directory.CreateDirectory("captcha");

            string fileName =
                (req.FileName ?? Guid.NewGuid().ToString());

            string path = Path.GetFullPath(Path.Combine("captcha", fileName));

            await File.WriteAllBytesAsync(path, bytes);

            Console.WriteLine("Saved: " + path);

            var pyResp = await http.PostAsJsonAsync(
                pythonApi,
                new { image_path = path }
            );

            if (!pyResp.IsSuccessStatusCode)
                return Results.Problem("API failed");

            var result =
                await pyResp.Content.ReadFromJsonAsync<PythonResponse>();

            sw.Stop();

            Console.WriteLine($"TEXT = {result!.prediction}");
            Console.WriteLine($"CONF = {result.confidence}");
            Console.WriteLine("=========================================\n");

            return Results.Ok(new
            {
                text = result.prediction,
                conf = result.confidence,
                ms = sw.ElapsedMilliseconds
            });
        });

        app.Run("http://127.0.0.1:5077");
    }
}

public record CaptchaRequest(
    [property: JsonPropertyName("base64")] string Base64,
    [property: JsonPropertyName("fileName")] string? FileName);

public class PythonResponse
{
    public string prediction { get; set; }
    public double confidence { get; set; }
}
