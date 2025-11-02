from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Backend activo en Vercel"}

@app.get("/meli/callback")
def meli_callback(code: str = None):
    # Aquí Mercado Libre redirige tras el login OAuth
    return JSONResponse({
        "status": "ok",
        "code_recibido": code
    })

print("running")
