from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import json
import gzip
import zlib
from collections import Counter, defaultdict
import heapq
import pickle
import struct
import mimetypes

def upload_file(request):
    """Serve the upload.html template at root URL"""
    return render(request, 'home.html')

# Improved Compression Algorithm Functions with filename preservation

def huffman_compress(data, original_filename):
    """Huffman coding compression with embedded metadata including filename"""
    if not data:
        return b''
    
    # Count frequency of each byte
    frequency = Counter(data)
    
    # Build Huffman tree
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Create encoding dictionary
    huffman_codes = {}
    if heap:
        for pair in heap[0][1:]:
            huffman_codes[pair[0]] = pair[1]
    
    # Handle single character case
    if len(huffman_codes) == 1:
        huffman_codes[list(huffman_codes.keys())[0]] = '0'
    
    # Encode data
    encoded_bits = ''.join(huffman_codes[byte] for byte in data)
    
    # Pad to make it byte-aligned
    padding = 8 - len(encoded_bits) % 8
    if padding != 8:
        encoded_bits += '0' * padding
    
    # Convert to bytes
    compressed_data = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte = encoded_bits[i:i+8]
        compressed_data.append(int(byte, 2))
    
    # Create metadata including original filename
    metadata = {
        'codes': huffman_codes,
        'padding': padding,
        'original_length': len(data),
        'original_filename': original_filename
    }
    
    # Serialize metadata and embed it in the file
    metadata_bytes = pickle.dumps(metadata)
    metadata_length = len(metadata_bytes)
    
    # File format: [metadata_length(4 bytes)][metadata][compressed_data]
    final_data = struct.pack('I', metadata_length) + metadata_bytes + bytes(compressed_data)
    
    return final_data

def huffman_decompress(compressed_file_data):
    """Huffman coding decompression with embedded metadata"""
    if len(compressed_file_data) < 4:
        raise ValueError("Invalid compressed file format")
    
    # Extract metadata length
    metadata_length = struct.unpack('I', compressed_file_data[:4])[0]
    
    # Extract metadata
    metadata_bytes = compressed_file_data[4:4+metadata_length]
    metadata = pickle.loads(metadata_bytes)
    
    # Extract compressed data
    compressed_data = compressed_file_data[4+metadata_length:]
    
    codes = metadata['codes']
    padding = metadata['padding']
    original_length = metadata['original_length']
    original_filename = metadata.get('original_filename', 'decompressed_file.txt')
    
    # Reverse the codes dictionary
    reverse_codes = {v: k for k, v in codes.items()}
    
    # Convert bytes to bits
    bits = ''.join(format(byte, '08b') for byte in compressed_data)
    
    # Remove padding
    if padding != 8:
        bits = bits[:-padding]
    
    # Decode
    decoded_data = bytearray()
    current_code = ""
    
    for bit in bits:
        current_code += bit
        if current_code in reverse_codes:
            decoded_data.append(reverse_codes[current_code])
            current_code = ""
            if len(decoded_data) >= original_length:
                break
    
    return bytes(decoded_data), original_filename

def rle_compress(data, original_filename):
    """Run-Length Encoding compression with filename preservation"""
    if not data:
        return b''
    
    compressed = bytearray()
    i = 0
    
    while i < len(data):
        current_byte = data[i]
        count = 1
        
        # Count consecutive identical bytes (max 255)
        while i + count < len(data) and data[i + count] == current_byte and count < 255:
            count += 1
        
        compressed.append(count)
        compressed.append(current_byte)
        i += count
    
    # Add filename metadata
    metadata = {'original_filename': original_filename}
    metadata_bytes = pickle.dumps(metadata)
    metadata_length = len(metadata_bytes)
    
    # File format: [metadata_length(4 bytes)][metadata][compressed_data]
    final_data = struct.pack('I', metadata_length) + metadata_bytes + bytes(compressed)
    
    return final_data

def rle_decompress(compressed_file_data):
    """Run-Length Encoding decompression with filename restoration"""
    if len(compressed_file_data) < 4:
        raise ValueError("Invalid compressed file format")
    
    # Extract metadata length
    metadata_length = struct.unpack('I', compressed_file_data[:4])[0]
    
    # Extract metadata
    metadata_bytes = compressed_file_data[4:4+metadata_length]
    metadata = pickle.loads(metadata_bytes)
    
    # Extract compressed data
    compressed_data = compressed_file_data[4+metadata_length:]
    original_filename = metadata.get('original_filename', 'decompressed_file.txt')
    
    decompressed = bytearray()
    
    for i in range(0, len(compressed_data), 2):
        if i + 1 < len(compressed_data):
            count = compressed_data[i]
            byte_value = compressed_data[i + 1]
            decompressed.extend([byte_value] * count)
    
    return bytes(decompressed), original_filename

def lz77_compress(data, original_filename, window_size=4096, lookahead_size=18):
    """LZ77 compression with filename preservation"""
    if not data:
        return b''
    
    compressed = []
    i = 0
    
    while i < len(data):
        match_length = 0
        match_distance = 0
        
        # Search for matches in the sliding window
        start = max(0, i - window_size)
        for j in range(start, i):
            length = 0
            while (i + length < len(data) and 
                   j + length < i and 
                   data[j + length] == data[i + length] and 
                   length < lookahead_size):
                length += 1
            
            if length > match_length:
                match_length = length
                match_distance = i - j
        
        if match_length > 0:
            # Store (distance, length, next_char)
            next_char = data[i + match_length] if i + match_length < len(data) else 0
            compressed.append((match_distance, match_length, next_char))
            i += match_length + 1
        else:
            # Store literal character
            compressed.append((0, 0, data[i]))
            i += 1
    
    # Convert to bytes
    compressed_bytes = bytearray()
    for distance, length, char in compressed:
        compressed_bytes.extend(distance.to_bytes(2, 'big'))
        compressed_bytes.append(length)
        compressed_bytes.append(char)
    
    # Add filename metadata
    metadata = {'original_filename': original_filename}
    metadata_bytes = pickle.dumps(metadata)
    metadata_length = len(metadata_bytes)
    
    # File format: [metadata_length(4 bytes)][metadata][compressed_data]
    final_data = struct.pack('I', metadata_length) + metadata_bytes + bytes(compressed_bytes)
    
    return final_data

def lz77_decompress(compressed_file_data):
    """LZ77 decompression with filename restoration"""
    if len(compressed_file_data) < 4:
        raise ValueError("Invalid compressed file format")
    
    # Extract metadata length
    metadata_length = struct.unpack('I', compressed_file_data[:4])[0]
    
    # Extract metadata
    metadata_bytes = compressed_file_data[4:4+metadata_length]
    metadata = pickle.loads(metadata_bytes)
    
    # Extract compressed data
    compressed_data = compressed_file_data[4+metadata_length:]
    original_filename = metadata.get('original_filename', 'decompressed_file.txt')
    
    decompressed = bytearray()
    i = 0
    
    while i < len(compressed_data):
        if i + 3 < len(compressed_data):
            distance = int.from_bytes(compressed_data[i:i+2], 'big')
            length = compressed_data[i+2]
            char = compressed_data[i+3]
            
            if distance > 0 and length > 0:
                # Copy from sliding window
                start = len(decompressed) - distance
                for j in range(length):
                    decompressed.append(decompressed[start + j])
            
            decompressed.append(char)
            i += 4
    
    return bytes(decompressed), original_filename

def gzip_compress(data, original_filename):
    """GZIP compression with filename preservation"""
    import io
    
    # Create in-memory buffer
    buffer = io.BytesIO()
    
    # Write gzip header with original filename
    with gzip.GzipFile(fileobj=buffer, mode='wb', filename=original_filename.encode('utf-8')) as gz_file:
        gz_file.write(data)
    
    compressed_data = buffer.getvalue()
    
    # Add our own metadata wrapper to ensure filename preservation
    metadata = {'original_filename': original_filename}
    metadata_bytes = pickle.dumps(metadata)
    metadata_length = len(metadata_bytes)
    
    # File format: [metadata_length(4 bytes)][metadata][gzip_compressed_data]
    final_data = struct.pack('I', metadata_length) + metadata_bytes + compressed_data
    
    return final_data

def gzip_decompress(compressed_file_data):
    """GZIP decompression with filename restoration"""
    import io
    
    # Check if this is our wrapped format or plain gzip
    if len(compressed_file_data) < 4:
        raise ValueError("Invalid compressed file format")
    
    try:
        # Try to extract our metadata first
        metadata_length = struct.unpack('I', compressed_file_data[:4])[0]
        
        # Validate metadata length is reasonable
        if metadata_length > 0 and metadata_length < len(compressed_file_data):
            # Extract metadata
            metadata_bytes = compressed_file_data[4:4+metadata_length]
            metadata = pickle.loads(metadata_bytes)
            
            # Extract gzip data
            gzip_data = compressed_file_data[4+metadata_length:]
            original_filename = metadata.get('original_filename', 'decompressed_file.txt')
            
            # Decompress the gzip data
            decompressed_data = gzip.decompress(gzip_data)
            
            return decompressed_data, original_filename
            
    except (struct.error, pickle.UnpicklingError, gzip.BadGzipFile):
        # If metadata extraction fails, treat as plain gzip file
        pass
    
    # Fallback: treat as plain gzip file
    try:
        # Try to extract filename from gzip header
        buffer = io.BytesIO(compressed_file_data)
        with gzip.GzipFile(fileobj=buffer, mode='rb') as gz_file:
            decompressed_data = gz_file.read()
            # Try to get original filename from gzip header
            if hasattr(gz_file, 'name') and gz_file.name:
                if isinstance(gz_file.name, bytes):
                    original_filename = gz_file.name.decode('utf-8', errors='ignore')
                else:
                    original_filename = str(gz_file.name)
            else:
                original_filename = 'decompressed_file.txt'
        
        return decompressed_data, original_filename
        
    except gzip.BadGzipFile:
        # Last resort: direct gzip decompress
        decompressed_data = gzip.decompress(compressed_file_data)
        return decompressed_data, 'decompressed_file.txt'

# Algorithm detection function
def detect_algorithm(filename):
    """Detect compression algorithm from filename"""
    if '_huffman.compressed' in filename:
        return 'huffman'
    elif '_rle.compressed' in filename:
        return 'rle'
    elif '_lz77.compressed' in filename:
        return 'lz77'
    elif '_gzip.compressed' in filename:
        return 'gzip'
    else:
        return 'gzip'  # Default fallback

# Main view functions
def compress_file(request):
    """Handle file compression"""
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES.get('file')
            algorithm = request.POST.get('algorithm', 'gzip')
            
            if not uploaded_file:
                return JsonResponse({'error': 'No file uploaded'}, status=400)
            
            # Read file data and get original filename
            file_data = uploaded_file.read()
            original_filename = uploaded_file.name
            
            # Choose compression algorithm
            if algorithm == 'huffman':
                compressed_data = huffman_compress(file_data, original_filename)
            elif algorithm == 'rle':
                compressed_data = rle_compress(file_data, original_filename)
            elif algorithm == 'lz77':
                compressed_data = lz77_compress(file_data, original_filename)
            elif algorithm == 'gzip':
                compressed_data = gzip_compress(file_data, original_filename)
            else:
                return JsonResponse({'error': 'Invalid algorithm'}, status=400)
            
            # Create response with compressed file
            response = HttpResponse(compressed_data, content_type='application/octet-stream')
            
            # Generate filename
            name_without_ext = os.path.splitext(original_filename)[0]
            compressed_filename = f"{name_without_ext}_{algorithm}.compressed"
            
            response['Content-Disposition'] = f'attachment; filename="{compressed_filename}"'
            response['X-Compression-Ratio'] = f"{len(compressed_data) / len(file_data):.2f}"
            response['X-Original-Size'] = str(len(file_data))
            response['X-Compressed-Size'] = str(len(compressed_data))
            response['X-Algorithm'] = algorithm
            response['X-Original-Filename'] = original_filename
            
            return response
            
        except Exception as e:
            return JsonResponse({'error': f'Compression failed: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)

def decompress_file(request):
    """Handle file decompression"""
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES.get('file')
            algorithm = request.POST.get('algorithm')
            
            if not uploaded_file:
                return JsonResponse({'error': 'No file uploaded'}, status=400)
            
            # Auto-detect algorithm from filename if not provided
            if not algorithm:
                algorithm = detect_algorithm(uploaded_file.name)
            
            # Read compressed file data
            compressed_data = uploaded_file.read()
            
            # Choose decompression algorithm and get original filename
            if algorithm == 'huffman':
                decompressed_data, original_filename = huffman_decompress(compressed_data)
            elif algorithm == 'rle':
                decompressed_data, original_filename = rle_decompress(compressed_data)
            elif algorithm == 'lz77':
                decompressed_data, original_filename = lz77_decompress(compressed_data)
            elif algorithm == 'gzip':
                decompressed_data, original_filename = gzip_decompress(compressed_data)
            else:
                return JsonResponse({'error': 'Invalid algorithm'}, status=400)
            
            # Determine content type based on original filename
            content_type, _ = mimetypes.guess_type(original_filename)
            if not content_type:
                content_type = 'application/octet-stream'
            
            # Create response with decompressed file
            response = HttpResponse(decompressed_data, content_type=content_type)
            response['Content-Disposition'] = f'attachment; filename="{original_filename}"'
            response['X-Original-Size'] = str(len(compressed_data))
            response['X-Decompressed-Size'] = str(len(decompressed_data))
            response['X-Algorithm'] = algorithm
            response['X-Original-Filename'] = original_filename
            
            return response
            
        except Exception as e:
            return JsonResponse({'error': f'Decompression failed: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)
