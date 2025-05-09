from src.linreg import LinearRegression

def main():
    model = LinearRegression(epochs=1000, lr=0.01)
    print(model)

if __name__ == '__main__':
    main()