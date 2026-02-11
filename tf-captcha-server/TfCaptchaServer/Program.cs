using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;
using System.Text.Json.Serialization;
using OpenCvSharp;
using Tensorflow;
using NumSharp;
using static Tensorflow.Binding;

Console.WriteLine("=== TF CAPTCHA SERVER (FROZEN GRAPH) ===");

const int IMG_W = 212;
const int IMG_H = 50;

const int MODEL_CLASSES = 66;

string chars =
"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@=#";

int blankIndex = chars.Length;

var graph = new Graph().as_default();
graph.Import(File.ReadAllBytes("captcha_tfnet_compatible.pb"));

var session = tf.Session(graph);

Console.WriteLine("Graph loaded");

string INPUT = "x";
string OUTPUT = "Identity";

float[] Preprocess(string path)
{
    var img = Cv2.ImRead(path, ImreadModes.Grayscale);

    Cv2.Resize(img, img, new Size(200, 50));

    Cv2.EqualizeHist(img, img);

    var padded = new Mat();
    Cv2.CopyMakeBorder(
        img, padded,
        top: 0, bottom: 0,
        left: 0, right: 12,
        borderType: BorderTypes.Constant,
        value: Scalar.White);

    padded.ConvertTo(padded, MatType.CV_32F, 1.0 / 255.0);

    float[] data = new float[IMG_H * IMG_W];

    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++)
            data[y * IMG_W + x] = padded.At<float>(y, x);

    return data;
}

float[] Run(float[] img)
{
    NDArray nd = np.array(img).reshape(1, IMG_H, IMG_W, 1);

    var result = session.run(
        graph.OperationByName(OUTPUT).output,
        new FeedItem(
            graph.OperationByName(INPUT).output,
            nd));

    return result.ToArray<float>();
}

(string text, double conf) Decode(float[] probs)
{
    int C = MODEL_CLASSES;
    int T = probs.Length / C;

    var charsOut = new List<char>();
    var confs = new List<float>();

    int prev = -1;

    for (int t = 0; t < T; t++)
    {
        int best = 0;
        float max = probs[t * C];

        for (int c = 1; c < C; c++)
        {
            float v = probs[t * C + c];
            if (v > max)
            {
                max = v;
                best = c;
            }
        }

        if (best == prev)
            continue;

        prev = best;

        if (best == blankIndex)
            continue;

        if (best < chars.Length)
        {
            charsOut.Add(chars[best]);
            confs.Add(max);
        }
    }

    double confidence = confs.Count == 0 ? 0 : confs.Average();
    return (new string(charsOut.ToArray()), confidence);
}

var builder = WebApplication.CreateBuilder();
var app = builder.Build();

app.MapPost("/predict_base64", async (CaptchaRequest req) =>
{
    try
    {
        var b64 = req.Base64;
        int comma = b64.IndexOf(',');
        if (comma > 0)
            b64 = b64[(comma + 1)..];

        var bytes = Convert.FromBase64String(b64);

        var path = "tmp.png";
        await File.WriteAllBytesAsync(path, bytes);

        var img = Preprocess(path);
        var logits = Run(img);

        var (text, conf) = Decode(logits);

        return Results.Ok(new
        {
            prediction = text,
            confidence = Math.Round(conf, 3),
            timesteps = logits.Length / MODEL_CLASSES
        });
    }
    catch (Exception e)
    {
        Console.WriteLine(e);
        return Results.Problem(e.Message);
    }
});

app.Run("http://0.0.0.0:8080");

public record CaptchaRequest(
    [property: JsonPropertyName("base64")] string Base64,
    [property: JsonPropertyName("file_name")] string? FileName);
