import os

NAME_DATE = {
    "goto": "1219",
    "kawai": "1115",
    "matumoto": "1128",
    "takahashi": "1220",
    "takahashi_jr": "0512",
    "taniguchi": "1107",
    "yoshikura": "1130",
    "taniguchi": "1107",
    "sato": "1115",
    "takamatu": "1130",
    "mori": "0712",
    "mori": "0723",
    "gosha": "0807",
    "asano": "0714",
    "togo": "1107",
    "takahashi_kazuya": "0516",
    "nakanishi": "0407",
    "miyano": "0723",
    "gobara": "0512",
    "kinoshita": "0801",
    "nakashimizu": "0512",
    "nishio": "0513",
    "kanda": "0807",
    "noda": "0714",
    "ikejima": "0714",
    "henmi": "0304",
    "henmi2": "0304",
    "kasahara": "0304",
    "maruyama": "0304",
    "miyabayashi": "0304",
    "patient1": "1001",
    "patient2": "1001",
    "patient3": "1001",
    "patient4": "1001",
    "patient5": "1001",
    "patient6": "1001",
    "patient7": "1109",
    "patient8": "1109",
    "patient9": "1109",
    "patient10": "1109",
}  # 今は後藤修論の時の被験者の名前と日付になっているが新たに計測した場合はここを変更する。


def select_name_and_date():
    # 名前と日付の辞書
    names_and_dates = NAME_DATE
    # 名前のリストを表示
    names_list = list(names_and_dates.keys())

    while True:
        print("0: 新規に入力する")
        for i, name in enumerate(names_list):
            print(f"{i + 1}: {name},{names_and_dates[name]}")
        try:
            # ユーザーに番号で選択させる
            choice = int(input("選択してください（番号）: "))

            # 選択された名前と日付を確認
            if choice < 0 or choice >= len(names_list) + 2:
                raise ValueError("番号がリストの範囲外です。")
            if choice == 0:
                selected_name = input("name:")
                selected_date = input("date:")
            else:
                selected_name = names_list[choice - 1]
                selected_date = names_and_dates[selected_name]
            return selected_name, selected_date

        except ValueError as e:
            print(f"エラー: {e}。もう一度入力してください。")
