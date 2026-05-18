#!/usr/bin/env wolframscript

(* TOPAS dose CSV -> paper-style spot-size and entrance/exit metrics.
   Units:
   - Spatial metrics (z_hat, sigma_x, sigma_y, dx/dy/dz): cm
   - Relative dose metrics (entrance/exit): %
*)

ClearAll[getArgValue, parseTopasCSV, weightedMoments2D, fitGaussian2D, usage, toNum];

usage[] := (
  Print["Usage:"];
  Print["  wolframscript -file scripts/extract_topas_metrics_mathematica.wl --csv <dose.csv> [--out <metrics.json>]"];
  Print[""];
  Print["Example:"];
  Print["  wolframscript -file scripts/extract_topas_metrics_mathematica.wl --csv runs/cases/E250_p14p90/dose.csv --out runs/cases/E250_p14p90/metrics_mathematica.json"];
);

getArgValue[name_String, default_] := Module[{idx},
  idx = FirstPosition[$ScriptCommandLine, name, Missing["NotFound"]];
  If[idx === Missing["NotFound"], default,
    If[idx[[1]] < Length[$ScriptCommandLine], $ScriptCommandLine[[idx[[1]] + 1]], default]
  ]
];

toNum[s_String] := ToExpression[
  StringReplace[
    StringTrim[s],
    RegularExpression["([+-]?(?:\\d*\\.\\d+|\\d+))(?:[eE]([+-]?\\d+))"] :> "$1*^$2"
  ]
];

parseTopasCSV[csvPath_String] := Module[
  {lines, header = <||>, axisHits, axis, dataLines, rows},
  lines = Import[csvPath, "Lines"];
  If[!ListQ[lines], Return[$Failed]];

  Do[
    If[StringStartsQ[StringTrim[line], "#"],
      axisHits = StringCases[
        StringTrim[line],
        RegularExpression["^#\\s*([XYZ])\\s+in\\s+(\\d+)\\s+bins\\s+of\\s+([0-9eE+\\-.]+)\\s+cm\\s*$"] :> {"$1", "$2", "$3"}
      ];
      If[Length[axisHits] > 0,
        axis = ToLowerCase[axisHits[[1, 1]]];
        header["n" <> axis] = ToExpression[axisHits[[1, 2]]];
        header["d" <> axis <> "_cm"] = N@ToExpression[axisHits[[1, 3]]];
      ];
    ],
    {line, lines}
  ];

  dataLines = Select[
    lines,
    (StringTrim[#] =!= "" && !StringStartsQ[StringTrim[#], "#"]) &
  ];

  rows = N @ Map[
    toNum /@ (StringTrim /@ Take[StringSplit[#, ","], 4]) &,
    dataLines
  ];

  <|
    "header" -> header,
    "rows" -> rows
  |>
];

weightedMoments2D[slice_?MatrixQ, xCoords_List, yCoords_List] := Module[
  {nx, ny, baseline, w, sumW, xGrid, yGrid, muX, muY, sx, sy},
  nx = Length[xCoords];
  ny = Length[yCoords];
  baseline = Min[slice];
  w = Map[Max[0.0, # - baseline] &, slice, {2}];
  sumW = Total[w, 2] // Total;
  If[sumW <= 0, w = Map[Max[0.0, #] &, slice, {2}]; sumW = Total[w, 2] // Total];
  If[sumW <= 0,
    Return[<|"muX" -> 0.0, "muY" -> 0.0, "sx" -> 1.0, "sy" -> 1.0|>];
  ];

  xGrid = Outer[Times, xCoords, ConstantArray[1.0, ny]];
  yGrid = Outer[Times, ConstantArray[1.0, nx], yCoords];

  muX = N[Total[w*xGrid, 2] // Total]/sumW;
  muY = N[Total[w*yGrid, 2] // Total]/sumW;
  sx = Sqrt[Max[0.0, N[Total[w*(xGrid - muX)^2, 2] // Total]/sumW]];
  sy = Sqrt[Max[0.0, N[Total[w*(yGrid - muY)^2, 2] // Total]/sumW]];

  <|"muX" -> muX, "muY" -> muY, "sx" -> sx, "sy" -> sy|>
];

fitGaussian2D[slice_?MatrixQ, xCoords_List, yCoords_List] := Module[
  {nx, ny, moments, amp0, off0, sx0, sy0, muX0, muY0, pts, fit, params, sx, sy},
  nx = Length[xCoords];
  ny = Length[yCoords];
  moments = weightedMoments2D[slice, xCoords, yCoords];

  muX0 = moments["muX"];
  muY0 = moments["muY"];
  sx0 = Max[moments["sx"], 10^-6];
  sy0 = Max[moments["sy"], 10^-6];
  off0 = N@Min[slice];
  amp0 = Max[10^-12, N@Max[slice] - off0];

  pts = Flatten[
    Table[
      {
        {xCoords[[i]], yCoords[[j]]},
        N@slice[[i, j]]
      },
      {i, 1, nx},
      {j, 1, ny}
    ],
    1
  ];

  fit = Quiet@Check[
    NonlinearModelFit[
      pts,
      off + amp*Exp[-0.5*(((x - muX)/Exp[lSx])^2 + ((y - muY)/Exp[lSy])^2)],
      {
        {amp, amp0},
        {muX, muX0},
        {muY, muY0},
        {lSx, Log[sx0]},
        {lSy, Log[sy0]},
        {off, off0}
      },
      {x, y}
    ],
    $Failed
  ];

  If[fit === $Failed,
    Return[<|"sigmaX_cm" -> sx0, "sigmaY_cm" -> sy0, "fitStatus" -> "fallback_weighted_moments"|>]
  ];

  params = fit["BestFitParameters"];
  sx = N@Exp[lSx /. params];
  sy = N@Exp[lSy /. params];

  <|"sigmaX_cm" -> sx, "sigmaY_cm" -> sy, "fitStatus" -> "nonlinear_2d_gaussian"|>
];

Module[
  {
    csvPath, outPath, parsed, header, rows, nx, ny, nz, dx, dy, dz, grid, r, ix, iy, iz,
    cx, cy, onAxis, depthIntegrated, peakOnAxisIdx, peakIntegratedIdx, peakGlobalPos,
    peakGlobalList,
    zHatOnAxisCm, zHatIntegratedCm, zHatGlobalCm, xCoords, yCoords,
    focalSlice, entranceSlice, exitSlice, focalFit, entranceFit, exitFit,
    globalPeak, entrancePlaneMaxPct, exitPlaneMaxPct, entranceOnAxisPct, exitOnAxisPct,
    result
  },
  csvPath = getArgValue["--csv", ""];
  outPath = getArgValue["--out", ""];

  If[csvPath === "" || !FileExistsQ[csvPath],
    usage[];
    Print["Error: --csv path is required and must exist."];
    Exit[1];
  ];

  parsed = parseTopasCSV[csvPath];
  If[parsed === $Failed,
    Print["Error: could not read file: ", csvPath];
    Exit[2];
  ];

  header = parsed["header"];
  rows = parsed["rows"];

  If[Length[rows] == 0,
    Print["Error: no numeric TOPAS rows found in: ", csvPath];
    Exit[3];
  ];

  nx = Lookup[header, "nx", Max[rows[[All, 1]]] + 1];
  ny = Lookup[header, "ny", Max[rows[[All, 2]]] + 1];
  nz = Lookup[header, "nz", Max[rows[[All, 3]]] + 1];
  dx = Lookup[header, "dx_cm", Missing["dx"]];
  dy = Lookup[header, "dy_cm", Missing["dy"]];
  dz = Lookup[header, "dz_cm", Missing["dz"]];

  If[!NumericQ[dx] || !NumericQ[dy] || !NumericQ[dz],
    Print["Error: could not parse dx/dy/dz in cm from header."];
    Exit[4];
  ];

  grid = ConstantArray[0.0, {nx, ny, nz}];
  Do[
    r = rows[[k]];
    ix = Round[r[[1]]] + 1;
    iy = Round[r[[2]]] + 1;
    iz = Round[r[[3]]] + 1;
    If[1 <= ix <= nx && 1 <= iy <= ny && 1 <= iz <= nz,
      grid[[ix, iy, iz]] = N[r[[4]]];
    ],
    {k, 1, Length[rows]}
  ];

  cx = Quotient[nx, 2] + 1;
  cy = Quotient[ny, 2] + 1;
  onAxis = grid[[cx, cy, All]];
  depthIntegrated = Total[grid, {1, 2}];

  peakOnAxisIdx = First@Ordering[onAxis, -1];
  peakIntegratedIdx = First@Ordering[depthIntegrated, -1];
  (* Find the first global-maximum voxel index in full 3D level. *)
  peakGlobalList = Position[grid, Max[grid], {3}, 1];
  peakGlobalPos = If[
    Length[peakGlobalList] > 0,
    First[peakGlobalList],
    {cx, cy, peakIntegratedIdx}
  ];

  zHatOnAxisCm = (peakOnAxisIdx - 0.5)*dz;
  zHatIntegratedCm = (peakIntegratedIdx - 0.5)*dz;
  zHatGlobalCm = (peakGlobalPos[[3]] - 0.5)*dz;

  xCoords = (Range[nx] - 0.5 - nx/2.0)*dx;
  yCoords = (Range[ny] - 0.5 - ny/2.0)*dy;

  focalSlice = grid[[All, All, peakIntegratedIdx]];
  entranceSlice = grid[[All, All, 1]];
  exitSlice = grid[[All, All, nz]];

  focalFit = fitGaussian2D[focalSlice, xCoords, yCoords];
  entranceFit = fitGaussian2D[entranceSlice, xCoords, yCoords];
  exitFit = fitGaussian2D[exitSlice, xCoords, yCoords];

  globalPeak = Max[grid];
  entrancePlaneMaxPct = If[globalPeak > 0, 100.0*Max[entranceSlice]/globalPeak, Indeterminate];
  exitPlaneMaxPct = If[globalPeak > 0, 100.0*Max[exitSlice]/globalPeak, Indeterminate];

  entranceOnAxisPct = If[Max[onAxis] > 0, 100.0*First[onAxis]/Max[onAxis], Indeterminate];
  exitOnAxisPct = If[Max[onAxis] > 0, 100.0*Last[onAxis]/Max[onAxis], Indeterminate];

  result = <|
    "input_csv" -> csvPath,
    "grid_shape" -> <|"nx" -> nx, "ny" -> ny, "nz" -> nz|>,
    "voxel_size_cm" -> <|"dx_cm" -> dx, "dy_cm" -> dy, "dz_cm" -> dz|>,
    "z_hat_cm" -> <|
      "integrated_xy" -> zHatIntegratedCm,
      "on_axis" -> zHatOnAxisCm,
      "global_max" -> zHatGlobalCm
    |>,
    "spot_size_sigma_cm" -> <|
      "at_z_hat_integrated" -> <|
        "sigma_x_cm" -> focalFit["sigmaX_cm"],
        "sigma_y_cm" -> focalFit["sigmaY_cm"],
        "fit_status" -> focalFit["fitStatus"]
      |>,
      "entrance_slice_z0" -> <|
        "sigma_x_cm" -> entranceFit["sigmaX_cm"],
        "sigma_y_cm" -> entranceFit["sigmaY_cm"],
        "fit_status" -> entranceFit["fitStatus"]
      |>,
      "exit_slice_z_end" -> <|
        "sigma_x_cm" -> exitFit["sigmaX_cm"],
        "sigma_y_cm" -> exitFit["sigmaY_cm"],
        "fit_status" -> exitFit["fitStatus"]
      |>
    |>,
    "relative_dose_pct" -> <|
      "entrance_plane_max_over_global_max_pct" -> entrancePlaneMaxPct,
      "exit_plane_max_over_global_max_pct" -> exitPlaneMaxPct,
      "entrance_on_axis_over_on_axis_peak_pct" -> entranceOnAxisPct,
      "exit_on_axis_over_on_axis_peak_pct" -> exitOnAxisPct
    |>
  |>;

  Print["--- TOPAS Mathematica Extraction ---"];
  Print["Input: ", csvPath];
  Print["z_hat_integrated_cm: ", NumberForm[result["z_hat_cm", "integrated_xy"], {10, 4}]];
  Print["sigma_x_cm @ z_hat_integrated: ", NumberForm[result["spot_size_sigma_cm", "at_z_hat_integrated", "sigma_x_cm"], {10, 4}]];
  Print["sigma_y_cm @ z_hat_integrated: ", NumberForm[result["spot_size_sigma_cm", "at_z_hat_integrated", "sigma_y_cm"], {10, 4}]];
  Print["Entrance dose (% of global max): ", NumberForm[result["relative_dose_pct", "entrance_plane_max_over_global_max_pct"], {10, 3}]];
  Print["Exit dose (% of global max): ", NumberForm[result["relative_dose_pct", "exit_plane_max_over_global_max_pct"], {10, 3}]];

  If[outPath =!= "",
    Export[outPath, Normal[result], "JSON"];
    Print["Wrote JSON: ", outPath];
  ];
];
