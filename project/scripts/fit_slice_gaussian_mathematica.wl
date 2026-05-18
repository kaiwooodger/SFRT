#!/usr/bin/env wolframscript

(* Fit an axis-aligned 2D Gaussian with offset to a single x-y dose slice CSV.
   Input CSV columns: x_cm, y_cm, dose
*)

ClearAll[getArgValue, usage];
ClearAll[toNum];

usage[] := (
  Print["Usage:"];
  Print["  wolframscript -file scripts/fit_slice_gaussian_mathematica.wl --slice <slice.csv> [--out <fit.json>]"];
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

Module[
  {slicePath, outPath, lines, parts, pts, xVals, yVals, doseVals, amp0, off0, w, wsum, muX0, muY0, sx0, sy0, fit, params, sx, sy, mx, my, result},
  slicePath = getArgValue["--slice", ""];
  outPath = getArgValue["--out", ""];

  If[slicePath === "" || !FileExistsQ[slicePath],
    usage[];
    Print["Error: --slice path is required and must exist."];
    Exit[1];
  ];

  lines = Import[slicePath, "Lines"];
  If[!ListQ[lines] || Length[lines] == 0,
    Print["Error: no rows found in slice CSV: ", slicePath];
    Exit[2];
  ];

  pts = Reap[
    Do[
      parts = StringTrim /@ StringSplit[StringTrim[line], ","];
      If[Length[parts] >= 3,
        Sow[{{N@toNum[parts[[1]]], N@toNum[parts[[2]]]}, N@toNum[parts[[3]]]}]
      ],
      {line, lines}
    ]
  ];
  pts = If[Length[pts] >= 2 && Length[pts[[2]]] >= 1, pts[[2, 1]], {}];
  If[Length[pts] == 0,
    Print["Error: no numeric rows found in slice CSV: ", slicePath];
    Exit[3];
  ];

  xVals = pts[[All, 1, 1]];
  yVals = pts[[All, 1, 2]];
  doseVals = pts[[All, 2]];

  off0 = N@Min[doseVals];
  amp0 = Max[10^-12, N@Max[doseVals] - off0];
  w = Map[Max[0.0, # - off0] &, doseVals];
  wsum = Total[w];
  If[wsum <= 0, w = Map[Max[0.0, #] &, doseVals]; wsum = Total[w]];
  If[wsum <= 0, w = ConstantArray[1.0, Length[doseVals]]; wsum = Total[w]];

  muX0 = N[Total[w*xVals] / wsum];
  muY0 = N[Total[w*yVals] / wsum];
  sx0 = Sqrt[Max[10^-12, N[Total[w*(xVals - muX0)^2] / wsum]]];
  sy0 = Sqrt[Max[10^-12, N[Total[w*(yVals - muY0)^2] / wsum]]];

  fit = Quiet@Check[
    TimeConstrained[
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
      20,
      $Failed
    ],
    $Failed
  ];

  If[fit === $Failed,
    result = <|
      "fit_status" -> "fallback_weighted_moments",
      "sigma_x_cm" -> sx0,
      "sigma_y_cm" -> sy0,
      "mu_x_cm" -> muX0,
      "mu_y_cm" -> muY0
    |>,
    params = fit["BestFitParameters"];
    sx = Quiet@Check[N@Exp[lSx /. params], Indeterminate];
    sy = Quiet@Check[N@Exp[lSy /. params], Indeterminate];
    mx = Quiet@Check[N[muX /. params], Indeterminate];
    my = Quiet@Check[N[muY /. params], Indeterminate];
    If[NumericQ[sx] && NumericQ[sy] && NumericQ[mx] && NumericQ[my],
      result = <|
        "fit_status" -> "nonlinear_2d_gaussian",
        "sigma_x_cm" -> sx,
        "sigma_y_cm" -> sy,
        "mu_x_cm" -> mx,
        "mu_y_cm" -> my
      |>,
      result = <|
        "fit_status" -> "fallback_weighted_moments",
        "sigma_x_cm" -> sx0,
        "sigma_y_cm" -> sy0,
        "mu_x_cm" -> muX0,
        "mu_y_cm" -> muY0
      |>
    ]
  ];

  If[outPath =!= "",
    Export[outPath, Normal[result], "JSON"];
    Print["Wrote JSON: ", outPath];
  ];
  Print["sigma_x_cm: ", NumberForm[result["sigma_x_cm"], {10, 5}]];
  Print["sigma_y_cm: ", NumberForm[result["sigma_y_cm"], {10, 5}]];
];
