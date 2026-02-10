from insightface.app import FaceAnalysis

# This will auto-download buffalo_l model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

print("✅ InsightFace buffalo_l model downloaded")
