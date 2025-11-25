export type Landmark = {
  name: string
  description: string
  funFact: string
}

export const LANDMARKS: Record<number, Landmark> = {
  0: {
    name: 'The Old Well',
    description:
      'The beating heart of campus and a nod to the Temple of Love at Versailles, the Old Well welcomed generations of Tar Heels since the 1890s.',
    funFact: 'Sip from the fountain on the first day of classes for a little extra GPA luck.',
  },
  1: {
    name: 'Wilson Library',
    description:
      'A Beaux-Arts masterpiece opened in 1929, Wilson houses the North Carolina Collection, Southern Folklife archives, and endless marble details.',
    funFact: 'Look up to spot the ornate dome modeled after Renaissance reading rooms.',
  },
  2: {
    name: 'Bell Tower',
    description:
      'Dedicated in 1931, the Morehead-Patterson Bell Tower rings across campus every quarter hour and glows Carolina Blue on special nights.',
    funFact: 'Seniors climb all 172 steps the week before commencement for good luck.',
  },
  3: {
    name: 'Kenan Memorial Stadium',
    description:
      'Wrapped by towering pines, Kenan has hosted Tar Heel football for nearly a century and now seats more than 50,000 fans.',
    funFact: 'The grounds crew paints fresh Carolina Blue lines before every home game.',
  },
  4: {
    name: 'Ackland Art Museum',
    description:
      'From European masters to cutting-edge installations, Ackland has curated 20,000+ works for students and the Chapel Hill community since 1958.',
    funFact: 'Keep an eye out for rotating exhibits from student curators every semester.',
  },
}

export const FALLBACK_LANDMARK: Landmark = {
  name: 'Carolina Landmark',
  description:
    'We could not find an exact match, but it still looks like a special corner of UNC. Try another angle or a closer shot for a better prediction.',
  funFact: 'Campus history is everywhere—sometimes the mystery is half the fun.',
}

