# Audit cumulativ2-16noiembrie

Aceeasi cohorta pentru rata observata si media estimata; numai modele prospective cu rezultat verificat. Lipsurile pot introduce bias. Datele sunt deja vazute, nu validare independenta.

```json
{
  "goal70": {
    "all": {
      "n": 359,
      "success": 307,
      "fail": 52,
      "mean_p": 0.8739204022522981,
      "brier": 0.1228374012567228
    },
    "analyst": {
      "n": 11,
      "success": 9,
      "fail": 2,
      "mean_p": 0.92915188271523,
      "brier": 0.16087392837480888
    }
  },
  "corners1h": {
    "all": {
      "n": 248,
      "success": 222,
      "fail": 26,
      "mean_p": 0.8910475068558456,
      "brier": 0.0957483130524746
    },
    "analyst": {
      "n": 10,
      "success": 10,
      "fail": 0,
      "mean_p": 0.9514509919431802,
      "brier": 0.003002078670379475
    }
  }
}
```

[Toate esecurile modelate](all_modeled_failures.csv). Semnal important:ambele manse Islanda-Grecia au ratat89/72 desi erau candidati analist; evenimente dependente, nu doua experimente independente. La cornere, Oviedo-Girona9 avea dispersie mare cunoscuta; nu eliminam retrospectiv succesele cu aceeasi proprietate. Solutiile si criteriile de validare sunt in[propuneri](PROPOSALS_FOR_REVIEW.md).

Precizare de acoperire: subcohorta analyst include numai randurile cu flag analyst_selected explicit prezent. Numarul11Gol70/10cornere nu certifica recuperarea tuturor preferintelor analistului din toate zilele si nu se compara direct cu totalul alegerilor utilizatorului.
