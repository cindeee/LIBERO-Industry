# generate_pool.py
def generate_site_pool(num_sites=40):
    print("      ")
    print(f"      ")
    print("      ")
    print("      ")
    
    for i in range(num_sites):
        # We set alpha (the 4th rgba value) to 0 so they are invisible until called
        xml_line = f'      <site name="groove_pool_{i}" type="box" size="0.01 0.01 0.01" pos="0 0 -10.0" quat="1 0 0 0" rgba="0.2 0.2 0.8 0" group="1"/>'
        print(xml_line)

if __name__ == "__main__":
    generate_site_pool(100)