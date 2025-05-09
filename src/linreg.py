class LinearRegression:
    def __init__(self, epochs: int = 100, lr: float = 0.1) -> None:
        self._epochs = epochs
        self._lr = lr

    def __str__(self):
        representation = 'LinearRegression: '
        for key, value in self.__dict__.items():
            representation += f'{key.replace('_', '')}={value}' + ', '
        representation = representation[:-2]
        return representation
