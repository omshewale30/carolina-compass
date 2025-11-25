import { useCallback, useState } from 'react'
import Home from './views/Home'
import Inference from './views/Inference'

type View = 'landing' | 'inference'

function App() {
  const [view, setView] = useState<View>('landing')
  const [isFading, setIsFading] = useState(false)

  const transitionTo = useCallback(
    (nextView: View) => {
      if (nextView === view || isFading) return
      setIsFading(true)
      setTimeout(() => {
        setView(nextView)
      }, 240)
      setTimeout(() => setIsFading(false), 480)
    },
    [view, isFading],
  )

  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-slate-50 to-white text-slate-900">
      <div
        className={`transition-opacity duration-500 ease-out ${isFading ? 'opacity-0' : 'opacity-100'}`}
      >
        {view === 'landing' ? (
          <Home onStart={() => transitionTo('inference')} />
        ) : (
          <Inference onBack={() => transitionTo('landing')} />
        )}
      </div>
    </div>
  )
}

export default App

