class resultadosFinalesExportacion:
    def __init__(self,metrics):
        self.metrics = metrics
    def save_metrics_to_csv(self, file_path="./resultados_metricas.csv"):
        import csv
        import os
        print("ya se ha guardado el metrics")
        # Verificamos si hay métricas
        if not self.metrics:
            print("No metrics to save.")
            return
        fieldnames = self.metrics[0].keys()

        # Guardamos las métricas en un CSV
        with open(file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for metric in self.metrics:
                writer.writerow(metric)

        print(f"Metrics saved to {os.path.abspath(file_path)}")


    def upload_csv_to_google_sheets(self, csv_path, sheet_url, creds_path="credenciales.json"):
        print("Iniciando carga de datos en Google Sheets...")
        try:
            import csv, gspread
            from oauth2client.service_account import ServiceAccountCredentials
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
            client = gspread.authorize(creds)
            print("Autenticación completada")
        except Exception as e:
            print(f"Error en autenticación: {e}")
            return

        try:
            spreadsheet = client.open_by_url(sheet_url)
            worksheet = spreadsheet.get_worksheet(0)
            print("Hoja abierta correctamente")
        except Exception as e:
            print(f"Error al abrir la hoja: {e}")
            return

        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                data = list(reader)

            worksheet.update("A1", data)
            print("Datos subidos correctamente a Google Sheets")
        except Exception as e:
            print(f"Error al subir los datos: {e}")