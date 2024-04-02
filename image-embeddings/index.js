import https from "https";
import fs from "fs";
import { pipeline } from "@xenova/transformers";

const dot = (a, b) => {
  let product = 0;
  for (let i = 0; i < a.length; i++) {
    product += a[i] * b[i];
  }
  return product;
};

const length = (a) => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * a[i];
  }
  return Math.sqrt(sum);
};

const cosineSimilarity = (a, b) => {
  return dot(a, b) / (length(a) * length(b));
};

async function getRandomDogImages() {
  const DOG_API_URL = "https://dog.ceo/api/breeds/image/random";

  for (let i = 0; i < 10; i++) {
    try {
      const res = await fetch(DOG_API_URL);
      const data = await res.json();

      https.get(data.message, (res) => {
        const imagePath = `./images/dog${i}.jpg`;

        const filePath = fs.createWriteStream(imagePath);
        res.pipe(filePath);
        filePath.on("finish", () => {
          filePath.close();
          console.log("Download Completed");
        });
      });
      console.log(data);
    } catch (error) {
      console.log(error);
    }
  }
}

async function getRandomCatsImages() {
  //https://cataas.com/cat
  const CAT_API_URL = "https://api.thecatapi.com/v1/images/search";

  for (let i = 0; i < 10; i++) {
    try {
      const res = await fetch(CAT_API_URL);
      const data = await res.json();

      https.get(data[0].url, (res) => {
        const imagePath = `./images/cat${i}.jpg`;

        const filePath = fs.createWriteStream(imagePath);
        res.pipe(filePath);
        filePath.on("finish", () => {
          filePath.close();
          console.log("Download Completed");
        });
      });
      console.log(data);
    } catch (error) {
      console.log(error);
    }
  }
}

async function main() {
  //   await getRandomDogImages();
  // await getRandomCatsImages();

  const image_feature_extractor = await pipeline(
    "image-feature-extraction",
    "Xenova/clip-vit-base-patch32"
  );
  const url =
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png";
  const short = await image_feature_extractor("./images/dog2.jpg");
  const cat = await image_feature_extractor("./images/cat.jpg");
  const dog = await image_feature_extractor("./images/dog.jpg");
  const tree = await image_feature_extractor("./images/tree.jpg");
  const tShirt = await image_feature_extractor("./images/t-shirt.jpg");

  console.log(cosineSimilarity(short.data, short.data));
  console.log(cosineSimilarity(short.data, cat.data));
  console.log(cosineSimilarity(short.data, dog.data));
  console.log(cosineSimilarity(short.data, tree.data));
  console.log(cosineSimilarity(short.data, tShirt.data));
}

main();
