using System;
using System.Diagnostics;
using System.Text.Json.Serialization;
using System.Linq;
using System.Collections.Generic;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using OpenCvSharp;

namespace dot_net_server;

public class Program
{
    static readonly string Chars =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@=#";

    const int W = 200;
    const int H = 50;

    const int TTA_RUNS = 12;
    static Random Rng = new(1234); // fixed seed for reproducibility

    public static void Main(string[] args)
    {
        Console.WriteLine("=== BOOT ===");
        Console.WriteLine("OpenCV: " + Cv2.GetVersionString());

        var opt = new Microsoft.ML.OnnxRuntime.SessionOptions();
        var session = new InferenceSession("captcha_ctc.onnx", opt);

        var meta = session.InputMetadata.First();
        Console.WriteLine($"ONNX input name: {meta.Key}");
        Console.WriteLine($"ONNX dims: {string.Join(",", meta.Value.Dimensions.ToArray())}");

        var builder = WebApplication.CreateBuilder(args);
        var app = builder.Build();

        app.MapPost("/api/captcha", (CaptchaRequest req) =>
        {
            Console.WriteLine("\n================ REQUEST ================");

            var sw = Stopwatch.StartNew();

            var b64 = req.Base64;
            int comma = b64.IndexOf(',');
            if (comma > 0) b64 = b64[(comma + 1)..];

            byte[] bytes = Convert.FromBase64String(b64);

            using var baseMat = PreprocessBase(bytes);

            var (text, conf) = TtaPredict(session, baseMat, TTA_RUNS);

            sw.Stop();

            Console.WriteLine($"FINAL TEXT = {text}");
            Console.WriteLine($"FINAL CONF = {conf}");
            Console.WriteLine("=========================================\n");

            return Results.Ok(new { text, conf, ms = sw.ElapsedMilliseconds });
        });

        app.Run("http://127.0.0.1:5077");
    }

    // ================= PREPROCESS =================

    static Mat PreprocessBase(byte[] bytes)
    {
        var src = Cv2.ImDecode(bytes, ImreadModes.Grayscale);

        var resized = new Mat();
        Cv2.Resize(src, resized, new Size(W,H));
        Cv2.EqualizeHist(resized, resized);

        Console.WriteLine("pixel sample:");
        for(int i=0;i<5;i++)
            Console.WriteLine(resized.At<byte>(0,i));

        src.Dispose();
        return resized;
    }

    // ================= TTA =================

    static Mat TtaVariant(Mat src)
    {
        var v = src.Clone();

        if (Rng.NextDouble() < 0.5)
            Cv2.Dilate(v, v, Cv2.GetStructuringElement(
                MorphShapes.Rect, new Size(2,2)));
        else
            Cv2.Erode(v, v, Cv2.GetStructuringElement(
                MorphShapes.Rect, new Size(2,2)));

        v.ConvertTo(v, MatType.CV_32F, 1.0/255);

        double alpha = 1.05;
        double beta  = 0.01;

        Cv2.Multiply(v, alpha, v);
        Cv2.Add(v, beta, v);
        Cv2.Min(v, 1.0, v);
        Cv2.Max(v, 0.0, v);

        return v;
    }

    // ================= TTA BATCH =================

    static (string,float) TtaPredict(
        InferenceSession session,
        Mat baseImg,
        int runs)
    {
        var mats = new List<Mat>();

        for(int i=0;i<runs;i++)
            mats.Add(TtaVariant(baseImg));

        float[] data = new float[runs*H*W];
        int k=0;

        foreach(var m in mats)
        {
            for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                data[k++] = m.At<float>(y,x);
        }

        Console.WriteLine("tensor checksum: " +
            data.Take(1000).Sum());

        Console.WriteLine("tensor sample:");
        for(int i=0;i<5;i++)
            Console.WriteLine(data[i]);

        var tensor = new DenseTensor<float>(
            data, new[]{runs,H,W,1});

        using var res = session.Run(
            new[]{ NamedOnnxValue.CreateFromTensor("image", tensor) });

        var t = res.First().AsTensor<float>();

        Console.WriteLine("output dims: " +
            string.Join(",", t.Dimensions.ToArray()));

        var texts = new List<string>();
        var confs = new List<float>();

        for(int b=0;b<runs;b++)
        {
            var (txt,cf) = DecodeOne(t,b);
            texts.Add(txt);
            confs.Add(cf);
            Console.WriteLine($"TTA[{b}] = {txt}  conf={cf}");
        }

        foreach(var m in mats) m.Dispose();

        var best = texts.GroupBy(x=>x)
                        .OrderByDescending(g=>g.Count())
                        .First().Key;

        float vote = texts.Count(x=>x==best)/(float)runs;
        float mean = confs.Zip(texts)
                          .Where(p=>p.Second==best)
                          .Select(p=>p.First)
                          .DefaultIfEmpty(0)
                          .Average();

        Console.WriteLine($"vote={vote} mean={mean}");

        return (best, 0.6f*vote + 0.4f*mean);
    }

    // ================= CTC =================

    static (string,float) DecodeOne(Tensor<float> t,int b)
    {
        int T = t.Dimensions[1];
        int C = t.Dimensions[2];
        int blank = Chars.Length;

        Console.WriteLine("top5 timestep0:");
        var top = Enumerable.Range(0,C)
            .Select(c => (c, t[b,0,c]))
            .OrderByDescending(p=>p.Item2)
            .Take(5);

        foreach(var p in top)
            Console.WriteLine($"class {p.c} prob {p.Item2}");

        var raw = new List<int>();
        var probs = new List<float>();

        for(int i=0;i<T;i++)
        {
            int best=0;
            float bestVal=float.MinValue;

            for(int c=0;c<C;c++)
            {
                float v=t[b,i,c];
                if(v>bestVal){bestVal=v;best=c;}
            }

            raw.Add(best);
            probs.Add(bestVal);
        }

        Console.WriteLine("raw first 10: " +
            string.Join(",", raw.Take(10)));

        var final = new List<int>();
        var finalProb = new List<float>();
        int prev=-1;

        for(int i=0;i<raw.Count;i++)
        {
            if(raw[i]!=prev)
            {
                if(raw[i]!=blank && raw[i]<Chars.Length)
                {
                    final.Add(raw[i]);
                    finalProb.Add(probs[i]);
                }
                prev=raw[i];
            }
        }

        string text = new string(final.Select(i=>Chars[i]).ToArray());
        float conf = finalProb.Count>0 ? finalProb.Average() : 0;

        Console.WriteLine("decoded=" + text);
        Console.WriteLine("conf=" + conf);

        return (text,conf);
    }
}

public record CaptchaRequest(
    [property: JsonPropertyName("base64")] string Base64,
    [property: JsonPropertyName("fileName")] string? FileName);
