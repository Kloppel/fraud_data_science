# Human Decision Tree Rules

## Leaf 1

- If: V74 <= 0 AND D3 <= 1 AND M4 == __missing__
- Predicted fraud probability: 0.3211
- Training samples: 17022
- Training fraud rate: 0.0169

## Leaf 2

- If: V74 <= 0 AND D3 <= 1 AND M4 != __missing__
- Predicted fraud probability: 0.6562
- Training samples: 39497
- Training fraud rate: 0.0647

## Leaf 3

- If: V74 <= 0 AND D3 > 1 AND M4 == __missing__
- Predicted fraud probability: 0.1467
- Training samples: 137239
- Training fraud rate: 0.0062

## Leaf 4

- If: V74 <= 0 AND D3 > 1 AND M4 != __missing__
- Predicted fraud probability: 0.3610
- Training samples: 137138
- Training fraud rate: 0.0201

## Leaf 5

- If: V74 > 0 AND C11 <= 1 AND V258 <= 1
- Predicted fraud probability: 0.4310
- Training samples: 67109
- Training fraud rate: 0.0267

## Leaf 6

- If: V74 > 0 AND C11 <= 1 AND V258 > 1
- Predicted fraud probability: 0.7445
- Training samples: 20325
- Training fraud rate: 0.0955

## Leaf 7

- If: V74 > 0 AND C11 > 1 AND C11 <= 2
- Predicted fraud probability: 0.7799
- Training samples: 12438
- Training fraud rate: 0.1138

## Leaf 8

- If: V74 > 0 AND C11 > 1 AND C11 > 2
- Predicted fraud probability: 0.9288
- Training samples: 12137
- Training fraud rate: 0.3212
