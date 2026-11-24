using System;
using System.IO;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

internal static class GarchBellman
{
    private static double[] ReadDoubles(BinaryReader reader, int size)
    {
        var data = new double[size];
        Buffer.BlockCopy(reader.ReadBytes(size * sizeof(double)), 0, data, 0, size * sizeof(double));
        return data;
    }

    private static int[] ReadIndices(BinaryReader reader, int size)
    {
        var data = new int[size];
        Buffer.BlockCopy(reader.ReadBytes(size * sizeof(int)), 0, data, 0, size * sizeof(int));
        return data;
    }

    // V(m,k) = k*M(m) + F(m,k), where M(m) is the expected terminal rate.
    // F is constant above the maximum quantity that can be executed before expiry.
    private static unsafe double[] Solve(int horizon, int tickets, int baseSize, int first, int last, int observed,
        double[] rates, double[] lastMean, int branches, int[] indices, double[] weights)
    {
        int states = rates.Length;
        int stride = (tickets + 5) / 4 * 4;
        var thresholds = new double[(horizon + 1) * (tickets + 1)];
        for (int i = 0; i < thresholds.Length; i++)
            thresholds[i] = double.NaN;
        var next = new double[states * stride];
        var current = new double[states * stride];
        var expected = new double[stride];

        int nextHigh = Math.Min(tickets, baseSize);
        for (int s = 0; s < states; s++)
        {
            next[s * stride] = lastMean[s];
            double savings = Math.Min(0, rates[s] - lastMean[s]);
            for (int k = 1; k <= nextHigh; k++)
                next[s * stride + 1 + k] = k * savings;
        }
        if (horizon - 1 <= last)
            for (int k = 1; k <= tickets; k++)
                thresholds[(horizon - 1) * (tickets + 1) + k] = lastMean[observed];

        fixed (int* indexBase = indices)
        fixed (double* weightBase = weights, exp = expected)
        {
            for (int m = horizon - 2; m >= first; m--)
            {
                int high = Math.Min(tickets, baseSize * (horizon - m));
                int width = nextHigh + 2;
                int vectors = (width + 3) / 4;
                bool record = m <= last;
                fixed (double* source = next, destination = current)
                {
                    for (int s = 0; s < states; s++)
                    {
                        int offset = s * branches;
                        if (Fma.IsSupported && vectors <= 6)
                        {
                            var v0 = Vector256<double>.Zero;
                            var v1 = Vector256<double>.Zero;
                            var v2 = Vector256<double>.Zero;
                            var v3 = Vector256<double>.Zero;
                            var v4 = Vector256<double>.Zero;
                            var v5 = Vector256<double>.Zero;
                            for (int i = offset; i < offset + branches; i++)
                            {
                                double* values = source + indexBase[i] * stride;
                                var weight = Vector256.Create(weightBase[i]);
                                v0 = Fma.MultiplyAdd(weight, Avx.LoadVector256(values), v0);
                                if (vectors >= 2)
                                    v1 = Fma.MultiplyAdd(weight, Avx.LoadVector256(values + 4), v1);
                                if (vectors >= 3)
                                    v2 = Fma.MultiplyAdd(weight, Avx.LoadVector256(values + 8), v2);
                                if (vectors >= 4)
                                    v3 = Fma.MultiplyAdd(weight, Avx.LoadVector256(values + 12), v3);
                                if (vectors >= 5)
                                    v4 = Fma.MultiplyAdd(weight, Avx.LoadVector256(values + 16), v4);
                                if (vectors >= 6)
                                    v5 = Fma.MultiplyAdd(weight, Avx.LoadVector256(values + 20), v5);
                            }
                            Avx.Store(exp, v0);
                            if (vectors >= 2)
                                Avx.Store(exp + 4, v1);
                            if (vectors >= 3)
                                Avx.Store(exp + 8, v2);
                            if (vectors >= 4)
                                Avx.Store(exp + 12, v3);
                            if (vectors >= 5)
                                Avx.Store(exp + 16, v4);
                            if (vectors >= 6)
                                Avx.Store(exp + 20, v5);
                        }
                        else
                        {
                            Array.Clear(expected, 0, width);
                            for (int i = offset; i < offset + branches; i++)
                            {
                                double* values = source + indexBase[i] * stride;
                                var weight = weightBase[i];
                                for (int c = 0; c < width; c++)
                                    exp[c] += weight * values[c];
                            }
                        }
                        double mean = exp[0];
                        double* target = destination + s * stride;
                        target[0] = mean;
                        target[1] = 0;
                        double rateDifference = rates[s] - mean;
                        for (int k = 1; k <= high; k++)
                        {
                            int quantity = Math.Min(k, baseSize);
                            int wait = 1 + Math.Min(k, nextHigh);
                            int trade = 1 + Math.Min(k - quantity, nextHigh);
                            target[1 + k] = Math.Min(exp[wait], quantity * rateDifference + exp[trade]);
                            if (record && s == observed)
                                thresholds[m * (tickets + 1) + k] = mean + (exp[wait] - exp[trade]) / quantity;
                        }
                        if (record && s == observed)
                            for (int k = high + 1; k <= tickets; k++)
                                thresholds[m * (tickets + 1) + k] = mean;
                    }
                }
                var swap = next;
                next = current;
                current = swap;
                nextHigh = high;
            }
        }
        return thresholds;
    }

    private static void Main()
    {
        using (var reader = new BinaryReader(Console.OpenStandardInput()))
        using (var writer = new BinaryWriter(Console.OpenStandardOutput()))
        {
            while (true)
            {
                int states;
                try { states = reader.ReadInt32(); }
                catch (EndOfStreamException) { return; }
                int branches = reader.ReadInt32();
                int observed = reader.ReadInt32();
                int tickets = reader.ReadInt32();
                int baseSize = reader.ReadInt32();
                int horizon = reader.ReadInt32();
                int first = reader.ReadInt32();
                int last = reader.ReadInt32();
                var rates = ReadDoubles(reader, states);
                var lastMean = ReadDoubles(reader, states);
                var indices = ReadIndices(reader, states * branches);
                var weights = ReadDoubles(reader, states * branches);
                var thresholds = Solve(horizon, tickets, baseSize, first, last, observed, rates, lastMean, branches, indices, weights);
                var response = new byte[thresholds.Length * sizeof(double)];
                Buffer.BlockCopy(thresholds, 0, response, 0, response.Length);
                writer.Write(response);
                writer.Flush();
            }
        }
    }
}



