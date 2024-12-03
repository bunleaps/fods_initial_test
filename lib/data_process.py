import xml.etree.ElementTree as ET
import csv

# Function to read and parse the XML file
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

# Function to process each conversation
def process_conversations(xml_root, predators_set):
    conversations_data = []

    for conversation in xml_root.findall('conversation'):
        combined_text = ""
        authors_set = set()

        # Combine all message texts and collect authors
        for message in conversation.findall('message'):
            author_id = message.find('author').text
            text = message.find('text').text

            # Ensure the text is not None before calling .strip()
            if text:
                combined_text += " " + text.strip()

            # Add author_id to the set if it's not None
            if author_id:
                authors_set.add(author_id.strip())

        # Debug: Print authors for this conversation
        print(f"Authors in conversation: {authors_set}")

        # Check if any author is in the predators list
        label = "safe"
        if authors_set & predators_set:  # If there's any intersection between authors and predators
            label = "harmful"
        
        # Debug: Print the label decision
        print(f"Label for conversation: {label}")

        # Add the conversation data to the list
        conversations_data.append([combined_text.strip(), label])

    return conversations_data

# Function to read predators from the text file
def read_predators(predators_file):
    with open(predators_file, 'r') as f:
        predators = set(line.strip() for line in f.readlines())
    return predators

# Function to write the data to a CSV file
def write_to_csv(conversations_data, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])  # Write header
        writer.writerows(conversations_data)

# Main function to run the script
def main():
    xml_file = 'pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'  # Update with your XML file path
    predators_file = 'pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'  # Update with predators file path
    output_csv = 'output_conversations.csv'  # Output CSV file

    # Read and parse the XML
    xml_root = parse_xml(xml_file)

    # Read the list of predators
    predators_set = read_predators(predators_file)

    # Debug: Print the predators set to verify
    print(f"Predators list: {predators_set}")

    # Process the conversations and get the labeled data
    conversations_data = process_conversations(xml_root, predators_set)

    # Write the processed data to a CSV file
    write_to_csv(conversations_data, output_csv)
    print(f"Data has been written to {output_csv}")

if __name__ == "__main__":
    main()
