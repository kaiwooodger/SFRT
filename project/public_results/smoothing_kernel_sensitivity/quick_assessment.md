# Smoothing-kernel sensitivity

- Cases tested: `case03, case04, case06`
- Smoothing kernels: `2.0 mm, 4.0 mm, 6.0 mm`

## Sigma 2.0 mm

- Biology-added brainstem failures: `0` / `3`
- Mean |no-sink minus physical| across primary endpoints: `190.308`
- Mean |with-sink minus no-sink| across primary endpoints: `0.382`
- Mean sink-to-biology shift ratio: `0.019`
- Conclusion at this kernel: `does not survive cleanly`

## Sigma 4.0 mm

- Biology-added brainstem failures: `2` / `3`
- Mean |no-sink minus physical| across primary endpoints: `9.265`
- Mean |with-sink minus no-sink| across primary endpoints: `0.651`
- Mean sink-to-biology shift ratio: `0.070`
- Conclusion at this kernel: `survives`

## Sigma 6.0 mm

- Biology-added brainstem failures: `2` / `3`
- Mean |no-sink minus physical| across primary endpoints: `10.243`
- Mean |with-sink minus no-sink| across primary endpoints: `0.672`
- Mean sink-to-biology shift ratio: `0.067`
- Conclusion at this kernel: `survives`
