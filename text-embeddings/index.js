import { pipeline } from "@xenova/transformers";

const dot = (a, b) => {
  let product = 0;
  for (let i = 0; i < a.length; i++) {
    product += a[i] * b[i];
  }
  return product;
};

async function main() {
  const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );
  const output1 = await extractor("Áo thun nam vải cotton", {
    pooling: "mean",
    normalize: true,
  });

  const output2 = await extractor("Áo polo nam màu đỏ", {
    pooling: "mean",
    normalize: true,
  });

  const output3 = await extractor("Mèo xanh đi phơi nắng", {
    pooling: "mean",
    normalize: true,
  });

  console.log("Dot Product:", dot(output1.data, output2.data));
  console.log("Dot Product:", dot(output1.data, output3.data));
}

main();
// Tensor {
//   type: 'float32',
//   data: Float32Array [0.09094982594251633, -0.014774246141314507, ...],
//   dims: [1, 384]
// }
