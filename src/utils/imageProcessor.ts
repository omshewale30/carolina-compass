import * as tf from '@tensorflow/tfjs'

const TARGET_SIZE = 224

export type ProcessedImage = {
  tensor: tf.Tensor4D
  previewUrl: string
}

const loadImageElement = (file: File) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const image = new Image()
    image.crossOrigin = 'anonymous'
    image.onload = () => {
      URL.revokeObjectURL(url)
      resolve(image)
    }
    image.onerror = (error) => {
      URL.revokeObjectURL(url)
      reject(error)
    }
    image.src = url
  })

export const prepareImageTensor = async (file: File): Promise<ProcessedImage> => {
  const imageElement = await loadImageElement(file)
  const previewUrl = URL.createObjectURL(file)

  const tensor = tf.tidy(() => {
    const pixels = tf.browser.fromPixels(imageElement)
    const resized = tf.image.resizeBilinear(pixels, [TARGET_SIZE, TARGET_SIZE], true)
    const normalized = resized.toFloat().div(127.5).sub(1)
    const batched = normalized.expandDims(0)
    pixels.dispose()
    return batched as tf.Tensor4D
  })

  return { tensor, previewUrl }
}

