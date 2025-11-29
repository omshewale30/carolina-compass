export type ProcessedImage = {
  file: File
  previewUrl: string
}

export const prepareImage = async (file: File): Promise<ProcessedImage> => {
  // Validate file type
  if (!file.type.startsWith('image/')) {
    throw new Error('File must be an image')
  }

  // Create preview URL
  const previewUrl = URL.createObjectURL(file)

  return { file, previewUrl }
}

