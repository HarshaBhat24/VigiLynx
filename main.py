import argparse
from phishing.randomf import PhishingDetector
from Malware.malCheck import VirusTotalScanner

class VigiLynx:
    def __init__(self):
        self.phishing = PhishingDetector()
        self.malware = VirusTotalScanner('api_key')

    def analyze(self, type, source):
        try:
            print('Analysis started')
            if type.lower() == "phishing":
                print('Analyzing phishing')
                self.phishing.predict_phishing(source)
                print('Phishing analysis completed')
            elif type.lower() == "malware":
                self.malware.scan_and_get_results(source)
            else:
                raise ValueError(f"Input {type} and/or {source} is not correct. \n Kindly refer to the documentation")
        except Exception as e:
            print(f"An error in VigiLynx: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', "--type", required=True, help="Type of analysis (phishing or malware)", type=str)
    parser.add_argument('-source', "--source", required=True, help="URL for phishing or file path for malware", type=str)
    args = parser.parse_args()
    type = args.type
    source = args.source
    analyzer = VigiLynx()
    analyzer.analyze(type, source)