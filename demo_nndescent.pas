{*******************************************************************************
  This source code is licensed under the MIT license found in the LICENSE file
  in the root directory of this source tree.

  This source code relies on the Faiss library wrapper for Pascal, compatible
  with Delphi and Lazarus environments.

  Access the library at: https://buylibs.com/pascal/faiss

*******************************************************************************}

program demo_nndescent;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, faiss;

var
  d, KK, nb, nq, k, recalls, qps, i, n, m: Integer;
  index: IndexNNDescentFlat;
  bruteforce: IndexFlat;
  database, queries, dis: array of Single;
  nns, gt_nns: array of idx_t;
  recall: Single;
  start, finish: TDateTime;
  secondsDiff: Int64;
begin
  // dimension of the vectors to index
  d := 64;
  KK := 64;

  // size of the database we plan to index
  nb := 10000;

  // make the index object and train it
  index := IndexNNDescentFlat.Create(d, KK, METRIC_L2);
  try
    index.nndescent.S := 10;
    index.nndescent.R := 32;
    index.nndescent.L := KK;
    index.nndescent.iter := 10;
    index.verbose := True;

    // generate labels by IndexFlat
    bruteforce := IndexFlat.Create(d, METRIC_L2);
    try
      SetLength(database, nb * d);
      for i := 0 to High(database) do
        database[i] := Random(MaxInt) / MaxInt * 1024;

      // populating the database
      index.Add(nb, database);
      bruteforce.Add(nb, database);

      nq := 1000;

      // searching the database
      WriteLn('Searching ...');
      index.nndescent.search_L := 50;

      SetLength(queries, nq * d);
      for i := 0 to High(queries) do
        queries[i] := Random(MaxInt) / MaxInt * 1024;

      k := 5;
      SetLength(nns, k * nq);
      SetLength(gt_nns, k * nq);
      SetLength(dis, k * nq);

      start := Now;
      index.Search(nq, queries, k, dis, nns);
      finish := Now;

      // find exact kNNs by brute force search
      bruteforce.Search(nq, queries, k, dis, gt_nns);

      recalls := 0;
      for i := 0 to nq - 1 do
        for n := 0 to k - 1 do
          for m := 0 to k - 1 do
            if nns[i * k + n] = gt_nns[i * k + m] then
              Inc(recalls);

      recall := recalls / (k * nq);
      secondsDiff := Round(finish - start);
      if secondsDiff = 0 then secondsDiff := 1;
      qps := Round(nq / secondsDiff);

      WriteLn(Format('Recall@%d: %f, QPS: %d', [k, recall, qps]));
    finally
      bruteforce.Free;
    end;
  finally
    index.Free;
  end;
end.
