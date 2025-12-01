type HomeProps = {
  onStart: () => void
}

const Home = ({ onStart }: HomeProps) => {
  return (
    <section className="relative flex min-h-screen items-center justify-center overflow-hidden bg-slate-950 text-white">
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(75,156,211,0.45),_transparent_55%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_bottom,_rgba(19,41,75,0.85),_rgba(8,15,30,0.95))]" />
        <div className="absolute inset-0 opacity-40 mix-blend-screen">
          <img
            src="https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&fit=crop&w=1400&q=60"
            alt="UNC Campus abstract"
            className="h-full w-full object-cover"
            loading="lazy"
          />
        </div>
        <div className="absolute inset-0 bg-gradient-to-b from-slate-950/40 via-slate-950/70 to-slate-950/90" />
      </div>

      <div className="relative z-10 flex max-w-5xl flex-col items-center gap-10 px-6 text-center md:px-12">
     
        <div className="flex flex-col gap-6">
          <h1 className="text-4xl font-semibold leading-tight text-white drop-shadow-md md:text-6xl">
            Carolina Compass
          </h1>
          <p className="mx-auto max-w-2xl text-lg text-slate-100 md:text-xl">
            Explore UNC&apos;s history through the lens of AI. Identify campus landmarks instantly and uncover
            the stories that shaped Chapel Hill.
          </p>
        </div>
        <button
          onClick={onStart}
          className="group inline-flex items-center gap-4 rounded-full bg-white/90 px-10 py-4 text-lg font-semibold text-slate-900 shadow-[0_20px_50px_rgba(15,23,42,0.35)] transition hover:bg-white"
        >
          Tell me what this is
          <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-[#4B9CD3] text-white transition group-hover:translate-x-1">
            →
          </span>
        </button>
        <div className="mt-16 flex flex-wrap items-center justify-center gap-8 text-left text-sm text-slate-200">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Powered by</p>
            <p className="text-base font-semibold text-white"> ViT B_16_imagenet1k</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Made for</p>
            <p className="text-base font-semibold text-white">UNC Chapel Hill Explorers</p>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Home

