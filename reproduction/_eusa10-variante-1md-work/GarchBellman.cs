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
    // F is constant once k reaches the number of discretionary dates left.
    // Balances below K - (m - 1) cannot occur from the original allocation.
    private static unsafe double[] Solve(int horizon, int tickets, int first, int last, int observed,
        double[] rates, double[] lastMean, int branches, int[] indices, double[] weights)
    {
        int states = rates.Length;
        int stride = (Math.Min(tickets + 1, horizon - tickets) + 4) / 4 * 4;
        var thresholds = new double[(horizon + 1) * (tickets + 1)];
        for (int i = 0; i < thresholds.Length; i++)
            thresholds[i] = double.NaN;
        var next = new double[states * stride];
        var current = new double[states * stride];
        var expected = new double[stride];

        int nextLow = Math.Max(0, tickets - (horizon - 2));
        int nextHigh = 1;
        for (int s = 0; s < states; s++)
        {
            next[s * stride] = lastMean[s];
            next[s * stride + 2 - nextLow] = Math.Min(0, rates[s] - lastMean[s]);
        }
        if (horizon - 1 <= last)
            for (int k = Math.Max(1, nextLow); k <= tickets; k++)
                thresholds[(horizon - 1) * (tickets + 1) + k] = lastMean[observed];

        fixed (int* indexBase = indices)
        fixed (double* weightBase = weights, exp = expected)
        {
            for (int m = horizon - 2; m >= first; m--)
            {
                int low = Math.Max(0, tickets - (m - 1));
                int high = Math.Min(tickets, horizon - m);
                int width = nextHigh - nextLow + 2;
                int vectors = (width + 3) / 4;
                bool record = m <= last;
                fixed (double* source = next, destination = current)
                {
                    for (int s = 0; s < states; s++)
                    {
                        int offset = s * branches;
                        if (Fma.IsSupported && vectors <= 3)
                        {
                            var v0 = Vector256<double>.Zero;
                            var v1 = Vector256<double>.Zero;
                            var v2 = Vector256<double>.Zero;
                            for (int i = offset; i < offset + branches; i++)
                            {
                                double* values = source + indexBase[i] * stride;
                                var weight = Vector256.Create(weightBase[i]);
                                v0 = Fma.MultiplyAdd(weight, Avx.LoadVector256(values), v0);
                                if (vectors >= 2)
                                    v1 = Fma.MultiplyAdd(weight, Avx.LoadVector256(values + 4), v1);
                                if (vectors == 3)
                                    v2 = Fma.MultiplyAdd(weight, Avx.LoadVector256(values + 8), v2);
                            }
                            Avx.Store(exp, v0);
                            if (vectors >= 2)
                                Avx.Store(exp + 4, v1);
                            if (vectors == 3)
                                Avx.Store(exp + 8, v2);
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
                        if (low == 0)
                            target[1] = 0;
                        double rateDifference = rates[s] - mean;
                        for (int k = Math.Max(1, low); k <= high; k++)
                        {
                            int wait = 1 + Math.Min(k, nextHigh) - nextLow;
                            int trade = 1 + Math.Min(k - 1, nextHigh) - nextLow;
                            target[1 + k - low] = Math.Min(exp[wait], rateDifference + exp[trade]);
                            if (record && s == observed)
                                thresholds[m * (tickets + 1) + k] = mean + (exp[wait] - exp[trade]);
                        }
                        if (record && s == observed)
                            for (int k = high + 1; k <= tickets; k++)
                                thresholds[m * (tickets + 1) + k] = mean;
                    }
                }
                var swap = next;
                next = current;
                current = swap;
                nextLow = low;
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
                int horizon = reader.ReadInt32();
                int first = reader.ReadInt32();
                int last = reader.ReadInt32();
                var rates = ReadDoubles(reader, states);
                var lastMean = ReadDoubles(reader, states);
                var indices = ReadIndices(reader, states * branches);
                var weights = ReadDoubles(reader, states * branches);
                var thresholds = Solve(horizon, tickets, first, last, observed, rates, lastMean, branches, indices, weights);
                var response = new byte[thresholds.Length * sizeof(double)];
                Buffer.BlockCopy(thresholds, 0, response, 0, response.Length);
                writer.Write(response);
                writer.Flush();
            }
        }
    }
}

