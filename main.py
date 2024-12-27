import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def cargar_datos(file_name: str) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_csv(file_name, encoding="latin1", delimiter=",")
    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Profit"] = pd.to_numeric(data["Profit"], errors="coerce")
    data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce")
    data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")
    data["Year"] = data["Order Date"].dt.year
    return data

def preparar_datos(data: pd.DataFrame) -> pd.DataFrame:
    data_ny: pd.DataFrame = data[data["City"] == "New York City"]
    data_ny["Profit"] = data_ny["Sales"] * (1 - data["Discount"])
    variables: pd.DataFrame = (
        data_ny.groupby("Year")
        .agg({"Sales": "sum", "Profit": "sum", "Discount": "mean", "Quantity": "sum"})
        .reset_index()
    )
    return variables

def entrenar_modelo(variables: pd.DataFrame, grado: int) -> tuple[Ridge, PolynomialFeatures, StandardScaler]:
    poly: PolynomialFeatures = PolynomialFeatures(degree=grado)
    X_poly: np.ndarray = poly.fit_transform(variables[["Year", "Profit", "Discount", "Quantity"]])
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)
    modelo: Ridge = Ridge(alpha=100.0)
    modelo.fit(X_poly_scaled, variables["Sales"])
    return modelo, poly, scaler

def predecir_ventas(modelo: Ridge, poly: PolynomialFeatures, scaler: StandardScaler, año: int, variables: pd.DataFrame) -> float:
    avg_profit: float = variables["Profit"].mean()
    avg_discount: float = variables["Discount"].mean()
    avg_quantity: float = variables["Quantity"].mean()
    X_2018: np.ndarray = poly.transform([[año, avg_profit, avg_discount, avg_quantity]])
    X_2018_scaled = scaler.transform(X_2018)
    ventas_pred: np.ndarray = modelo.predict(X_2018_scaled)
    return ventas_pred[0]

def predecir_profit(ventas_2018_pred: float, avg_discount: float) -> float:
    profit_2018_pred: float = ventas_2018_pred * (1 - avg_discount)
    return profit_2018_pred

def graficar_resultados(
    variables: pd.DataFrame, 
    modelo: Ridge, 
    poly: PolynomialFeatures, 
    scaler: StandardScaler, 
    ventas_2018_pred: float, 
    profit_2018_pred: float,
    grado: int
) -> None:
    plt.figure(figsize=(12, 6))
    plt.grid(True)   

    plt.scatter(
        variables["Year"], variables["Sales"], color="royalblue", label="Datos historicos de ventas", s=60, alpha=0.7
    )
    plt.scatter(
        variables["Year"], variables["Profit"], color="orange", label="Datos historicos de profit", s=60, alpha=0.7
    )

    years: np.ndarray = np.sort(variables["Year"].unique())
    avg_profit: float = variables["Profit"].mean()
    avg_discount: float = variables["Discount"].mean()
    avg_quantity: float = variables["Quantity"].mean()
    years_poly: np.ndarray = poly.transform(
        np.column_stack(
            [
                years,
                [avg_profit] * len(years),
                [avg_discount] * len(years),
                [avg_quantity] * len(years),
            ]
        )
    )
    years_poly_scaled = scaler.transform(years_poly)
    sales_pred: np.ndarray = modelo.predict(years_poly_scaled)

    plt.plot(years, sales_pred, color="red", label=f"Modelo de regresión polinomica (grado {grado})", linewidth=2)

    plt.scatter(2018, ventas_2018_pred, color="green", label="Predicción ventas 2018", zorder=5, s=100)
    plt.scatter(2018, profit_2018_pred, color="purple", label="Predicción profit 2018", zorder=5, s=100)

    plt.text(2018, ventas_2018_pred + 500, f"${ventas_2018_pred:,.2f}", ha="center", fontsize=12, color="green")
    plt.text(2018, profit_2018_pred + 500, f"${profit_2018_pred:,.2f}", ha="center", fontsize=12, color="purple")

    plt.title("Predicción de ventas y profit en New York", fontsize=14)
    plt.xlabel("Año", fontsize=12)
    plt.ylabel("Ventas y Profit ($)", fontsize=12)

    plt.xticks(np.arange(2014, 2020, 1))
    plt.yticks(np.arange(0, max(variables["Sales"].max(), ventas_2018_pred) + 5000, 5000))

    plt.legend(
        loc='lower right',
        fontsize=10, 
        bbox_to_anchor=(1, 0),
        borderpad=1,
        borderaxespad=1
    )
    plt.savefig("resultados_prediccion.png")    
    plt.show()
    


def main() -> None:
    file_name: str = "stores_sales_forecasting.csv" 
    data: pd.DataFrame = cargar_datos(file_name)
    variables: pd.DataFrame = preparar_datos(data)

    grado: int = 4
    modelo, poly, scaler = entrenar_modelo(variables, grado) 

    ventas_2018_pred: float = predecir_ventas(modelo, poly, scaler, 2018, variables) 
    print(f"Prediccion ventas NY 2018: ${ventas_2018_pred:,.2f}")
    
    avg_discount: float = variables["Discount"].mean() 
    profit_2018_pred: float = predecir_profit(ventas_2018_pred, avg_discount) 
    print(f"Prediccion Profit NY 2018: ${profit_2018_pred:,.2f}") 
    
    X_poly: np.ndarray = poly.fit_transform(variables[["Year", "Profit", "Discount", "Quantity"]]) 
    X_poly_scaled = scaler.transform(X_poly) 
    r_squared: float = modelo.score(X_poly_scaled, variables["Sales"])
    print(f"R² del modelo: {r_squared:.4f}")
    print("\n")
    print(variables.head())

    graficar_resultados(variables, modelo, poly, scaler, ventas_2018_pred, profit_2018_pred, grado)
    
if __name__ == "__main__":
    main()
