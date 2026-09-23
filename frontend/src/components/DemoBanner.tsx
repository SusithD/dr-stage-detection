export function DemoBanner() {
  return (
    <p className="demo-banner" role="status">
      Demo mode: no trained model is loaded, so results are simulated. Start the server with
      --checkpoint to use the trained model.
    </p>
  )
}
