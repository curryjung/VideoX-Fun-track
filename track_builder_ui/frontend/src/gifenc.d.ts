declare module "gifenc" {
  export function GIFEncoder(options?: { auto?: boolean }): {
    writeFrame(
      indexed: Uint8Array,
      width: number,
      height: number,
      options?: { palette?: number[][]; delay?: number; repeat?: number; transparent?: number | null }
    ): void;
    finish(): void;
    bytes(): Uint8Array;
  };
  export function quantize(rgba: Uint8ClampedArray, maxColors: number, options?: { format?: string }): number[][];
  export function applyPalette(rgba: Uint8ClampedArray, palette: number[][], format?: string): Uint8Array;
}
