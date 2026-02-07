using System.Diagnostics;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

string? customDir = Environment.GetEnvironmentVariable("CAPTCHA_SAVE_DIR");

string saveDir;

if (!string.IsNullOrWhiteSpace(customDir))
{
    saveDir = customDir;
}
else if (OperatingSystem.IsWindows())
{
    saveDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
        "captchas"
    );
}
else
{
    saveDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        "captchas"
    );
}

Directory.CreateDirectory(saveDir);
Console.WriteLine($"Captcha save directory: {saveDir}");

builder.WebHost.ConfigureKestrel(o =>
{
    o.Limits.MaxConcurrentConnections = 1000;
    o.Limits.MaxConcurrentUpgradedConnections = 1000;
    o.Limits.MaxRequestBodySize = 5 * 1024 * 1024; // 5MB
});

var app = builder.Build();

app.MapGet("/health", () => Results.Ok(new
{
    ok = true,
    service = "captcha-receiver",
    time = DateTime.UtcNow
}));

app.MapPost("/api/captcha", async (CaptchaRequest req) =>
{
    var sw = Stopwatch.StartNew();

    if (string.IsNullOrWhiteSpace(req.Base64))
        return Results.BadRequest("base64 missing");

    try
    {
        var b64 = req.Base64;
        int comma = b64.IndexOf(',');
        if (comma > 0)
            b64 = b64[(comma + 1)..];

        byte[] bytes = Convert.FromBase64String(b64);

        string name = req.FileName;
        if (string.IsNullOrWhiteSpace(name))
            name = Guid.NewGuid().ToString("N") + ".png";

        name = Path.GetFileName(name);

        string path = Path.Combine(saveDir, name);

        await File.WriteAllBytesAsync(path, bytes);

        sw.Stop();

        return Results.Ok(new
        {
            ok = true,
            savedAs = path,
            bytes = bytes.Length,
            ms = sw.ElapsedMilliseconds
        });
    }
    catch (FormatException)
    {
        return Results.BadRequest("invalid base64");
    }
    catch (Exception ex)
    {
        return Results.Problem(ex.Message);
    }
});

app.Run("http://127.0.0.1:5077");

record CaptchaRequest(
    [property: JsonPropertyName("base64")] string Base64,
    [property: JsonPropertyName("fileName")] string? FileName
);