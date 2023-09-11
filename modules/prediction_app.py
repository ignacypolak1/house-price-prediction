from modules import prediction_module
import tkinter as tk

class Panel(tk.Frame):
    def __init__(self,
                 parent,
                 label_text,
                 font,
                 **kwargs):
        super().__init__(master=parent, **kwargs)
        
        self.label = tk.Label(self, text=label_text, font=font)
        self.label.grid(row=0, column=0, sticky='nwse')

        self.entry = tk.Entry(self, font=font)
        self.entry.grid(row=1, column=0, sticky='nwse')

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
    
    def get_text(self) -> str:
        """Funkcja zwracająca wartość pola tekstowego obiektu entry

        Returns:
            str: Wartość pola tekstowego, czyli obiektu entry.
        """
        return self.entry.get()
                   
class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.root.title("Predykcja ceny")

        self.info_frame = tk.Frame(self.root)
        self.panels = {}

        for index, name in enumerate(prediction_module.ATTRIBUTES):
            self.panels[name] = Panel(self.info_frame, name, font=('Arial', 12), relief='sunken', borderwidth=1)
            self.panels[name].grid(row=index//4, column=index%4, padx=5, pady=5, sticky='nswe')

        for i in range(4):
            self.root.columnconfigure(i, weight=1)
            self.root.rowconfigure(i, weight=1)
            self.info_frame.columnconfigure(i, weight=1)
            self.info_frame.rowconfigure(i, weight=1)

        self.info_frame.grid(row=0, column=0, columnspan=4, sticky='nwse')

        self.button = tk.Button(self.root, text='Potwierdź', font=('Arial', 20), relief='groove', borderwidth=2, command=self.predict)
        self.button.grid(row=4, column=0, columnspan=4, sticky='nswe', pady=20)

        self.result_panel = Panel(self.root, "Przewidziana wartość:", font=('Arial', 16), relief='groove', borderwidth=1)
        self.result_panel.entry.configure(justify='center')
        self.result_panel.grid(row=5, column=0, columnspan=4, sticky='nswe')
    
    def run(self) -> None:
        """Funkcja uruchamiająca aplikację okienkową
        """
        self.root.mainloop()
    
    def predict(self) -> None:
        """Funkcja dokonująca predykcji w opraciu o model uczenia maszynowego
        """
        try:
            self.result_panel.entry.delete(0, tk.END)

            user_dict = {}

            for key, attribute in zip(prediction_module.KEYS, prediction_module.ATTRIBUTES):
                if '(m\N{SUPERSCRIPT TWO})' in attribute:
                    user_dict[key] = prediction_module.validate_numeric(self.panels, key=attribute, nonnegative=True, num_type='float')
                elif 'geograf' in attribute:
                    user_dict[key] = prediction_module.validate_numeric(self.panels, key=attribute, nonnegative=False, num_type='float')
                elif attribute == 'Czy nad nabrzeżem':
                    user_dict[key] = prediction_module.validate_numeric(self.panels, key=attribute, nonnegative=True, num_type='int', boolean=True)  
                elif attribute == 'Kod pocztowy':
                    user_dict[key] = prediction_module.validate_category(self.panels, key=attribute)
                else:
                    user_dict[key] = prediction_module.validate_numeric(self.panels, key=attribute, nonnegative=True, num_type='int')
    

            user_df = prediction_module.create_df_from_dict(user_dict)
            user_csr = prediction_module.process_data(user_df)
            predicted_value = prediction_module.predict_value(user_csr)
            
            self.result_panel.entry.insert(0, str(predicted_value) + '$')

        except Exception as e:
            self.result_panel.entry.insert(0, e)


