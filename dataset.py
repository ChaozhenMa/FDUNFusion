import os
from PIL import Image
from torch.utils.data import Dataset

class InfraredVisibleDataset(Dataset):
    def __init__(self, ir_dir, vis_dir, transform=None, image_size=128):
        """
        Infrared and Visible Image Fusion Dataset
        ir_dir: Infrared image directory path
        vis_dir: Visible image directory path
        """
        self.transform = transform
        self.image_size = image_size
        self.image_pairs = self.load_image_pairs(ir_dir, vis_dir)

        if len(self.image_pairs) == 0:
            raise ValueError(f"No matching image pairs found in directories {ir_dir} and {vis_dir}")

    def load_image_pairs(self, ir_dir, vis_dir):
        """Load matching infrared and visible image pairs - use natural sorting"""
        image_pairs = []

        def natural_sort_key(filename):
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith(valid_exts)], key=natural_sort_key)
        vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith(valid_exts)], key=natural_sort_key)

        if len(ir_files) == len(vis_files):
            for i in range(len(ir_files)):
                image_pairs.append((os.path.join(ir_dir, ir_files[i]), os.path.join(vis_dir, vis_files[i])))
        else:
            for ir_file in ir_files:
                vis_file = self.find_matching_file(ir_file, vis_files)
                if vis_file:
                    image_pairs.append((os.path.join(ir_dir, ir_file), os.path.join(vis_dir, vis_file)))
                else:
                    print(f"⚠️ No matching visible image found: {ir_file}")

        print(f"Number of successfully matched image pairs: {len(image_pairs)}")
        return image_pairs

    def find_matching_file(self, ir_file, vis_files):
        """Intelligent matching strategy supporting multiple filename formats"""
        ir_name, ir_ext = os.path.splitext(ir_file)

        # Strategy 1: Directly match the same filename
        if ir_file in vis_files: return ir_file

        # Strategy 2: IR/VIS format matching (IR1.tif <-> VIS1.tif)
        if ir_name.startswith('IR'):
            number_part = ir_name[2:]
            if f'VIS{number_part}{ir_ext}' in vis_files: return f'VIS{number_part}{ir_ext}'
            for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']:
                if f'VIS{number_part}{ext}' in vis_files: return f'VIS{number_part}{ext}'

        # Strategy 3: Visible to infrared format matching (VIS1.tif <-> IR1.tif)
        if ir_name.startswith('VIS'):
            number_part = ir_name[3:]
            if f'IR{number_part}{ir_ext}' in vis_files: return f'IR{number_part}{ir_ext}'

        # Strategy 4: Extract numbers for matching
        ir_numbers = self.extract_numbers(ir_name)
        if ir_numbers:
            for vis_file in vis_files:
                if ir_numbers == self.extract_numbers(os.path.splitext(vis_file)[0]):
                    return vis_file
        return None

    def extract_numbers(self, filename):
        import re
        numbers = re.findall(r'\d+', filename)
        return ''.join(numbers) if numbers else ''

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        ir_path, vis_path = self.image_pairs[idx]
        U = Image.open(ir_path).convert('RGB')
        V = Image.open(vis_path).convert('RGB')

        if self.transform:
            U = self.transform(U)
            V = self.transform(V)
        return U, V