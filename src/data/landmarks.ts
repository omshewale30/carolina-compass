export type Landmark = {
  name: string
  description: string
  funFact: string
}

export const LANDMARKS: Record<number, Landmark> = {
  0: {
    name: 'The Bell Tower',
    description:
      'Dedicated in 1931, the Morehead-Patterson Bell Tower rings across campus every quarter hour and glows Carolina Blue on special nights.',
    funFact: 'Seniors climb all 172 steps the week before commencement for good luck.',
  },
  1: {
    name: 'Gerrard Hall',
    description:
      'A Beaux-Arts masterpiece opened in 1929, Wilson houses the North Carolina Collection, Southern Folklife archives, and endless marble details.',
    funFact: 'Look up to spot the ornate dome modeled after Renaissance reading rooms.',
  },
  2: {
    name: 'Graham Hall',
    description:
      'Built in 1914, Graham Hall houses the Department of Political Science and houses a collection of rare books and manuscripts.',
    funFact: 'The hall is named after William Graham, a former UNC chancellor.',

  },
  3: {
    name: 'Person Hall',
    description:
      'Person Hall is a historic building on the University of North Carolina at Chapel Hill campus. It is named after Thomas Person, a former UNC chancellor.',
    funFact: 'The hall is home to the Department of Psychology and the Department of Sociology.',
  },
  4: {
    name: 'South Building',
    description:
      'The South Building is a historic building on the University of North Carolina at Chapel Hill campus. It is named after Thomas Person, a former UNC chancellor.',
    funFact: 'The building is home to the Department of Chemistry and the Department of Physics.',
  },
}



export const FALLBACK_LANDMARK: Landmark = {
  name: 'Carolina Landmark',
  description:
    'We could not find an exact match, but it still looks like a special corner of UNC. Try another angle or a closer shot for a better prediction.',
  funFact: 'Campus history is everywhere—sometimes the mystery is half the fun.',
}

