import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

final ThemeData myTheme = ThemeData(
  primaryColor: Color.fromARGB(255, 53, 43, 33),
  // scaffoldBackgroundColor: Colors.white,
  elevatedButtonTheme: ElevatedButtonThemeData(
    style: ButtonStyle(
        backgroundColor:
            MaterialStateProperty.all<Color>(Color.fromARGB(255, 82, 74, 31)),
        foregroundColor: MaterialStateProperty.all<Color>(const Color.fromARGB(255, 224, 206, 181))),
  ),
);

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle(
      statusBarColor: myTheme.primaryColor,
      statusBarBrightness: Brightness.light,
    ));
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'klasifikasi alat makan',
      theme: myTheme,
      home: const ImageUploader(),
    );
  }
}

class ImageUploader extends StatefulWidget {
  const ImageUploader({super.key});

  @override
  State<ImageUploader> createState() => _ImageUploaderState();
}

class _ImageUploaderState extends State<ImageUploader> {
  File? _image;
  bool isUploading = false;
  bool getResult = false;
  String responseMessage = '';

  Future<void> _setImage(ImageSource source) async {
    try {
      final image = await ImagePicker().pickImage(source: source);

      if (image == null) throw ("Tidak ada gambar terpilih.");

      setState(() {
        _image = File(image.path);
        responseMessage = '';
        getResult = false;
      });
    } catch (e) {
      throw ('Gagal memilih gambar: $e');
    }
  }

  Future captureImage() async {
    await _setImage(ImageSource.camera);
  }

  Future getImage() async {
    await _setImage(ImageSource.gallery);
  }

  Future uploadImage() async {
    try {
      if (_image == null) throw ("Tidak ada gambar terpilih.");

      setState(() {
        isUploading = true;
      });

      final uri = Uri.parse("http://192.168.0.114:5000/upload");
      var request = http.MultipartRequest('POST', uri);
      request.files
          .add(await http.MultipartFile.fromPath('image', _image!.path));
      var response = await request.send();

      if (response.statusCode != 200) {
        throw ("Tidak dapat terhubung dengan server.");
      }

      print('Gambar terupload');
      String responseBody = await response.stream.bytesToString();
      setState(() {
        responseMessage = json.decode(responseBody);
        getResult = true;
      });
    } catch (error) {
      print('Gagal mengupload gambar: $error');
    } finally {
      setState(() {
        isUploading = false;
      });
    }
  }

  void resetImage() {
    setState(() {
      _image = null;
      responseMessage = '';
      isUploading = false;
      getResult = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text('klasifikasi alat makan',
            style: TextStyle(
              color: Colors.white,
            )),
        backgroundColor: myTheme.primaryColor,
        actions: [
          IconButton(
            onPressed: resetImage,
            icon: const Icon(Icons.refresh),
            color: Colors.white,
          ),
        ],
      ),
      body: SafeArea(
        child: Stack(children: [
          Column(
            children: [
              Container(

                height: 300,
                child: _image != null
                    ? Image.file(
                        _image!,
                        fit: BoxFit.fitWidth,
                      )
                    : const Center(child: Text('Input gambar terlebuh dahulu')),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 10),
                child: Column(
                  children: [
                Visibility(
                        visible: _image != null && !isUploading,
                        child: CustomButton(
                          title: 'Upload ',
                          icon: Icons.upload,
                          onClick: uploadImage,
                        )),  
                    const SizedBox(height: 250),
                    CustomButton(
                      title: 'Galeri',
                      icon: Icons.photo,
                      onClick: getImage,
                    ),    
                    CustomButton(
                      title: 'Ambil Foto',
                      icon: Icons.camera_alt,
                      onClick: captureImage,
                    ),
                    Visibility(
                      visible: isUploading,
                      child: const CircularProgressIndicator(),
                    ),
                    const SizedBox(height: 30),
                  ],
                ),
              ),
            ],
          ),
          Visibility(
            visible: getResult,
            child: Positioned(
              bottom: 0,
              child: Container(
                height: 470,
                padding: const EdgeInsets.only(top: 24),
                width: MediaQuery.of(context).size.width,
                decoration: BoxDecoration(
                    color: Theme.of(context).primaryColor,
                  ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      'Di klasifikasikan sebagai',
                      style: TextStyle(fontSize: 16.0, color: Colors.white),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      responseMessage,
                      style:
                          const TextStyle(fontSize: 30.0, color: Colors.white),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ]),
      ),
    );
  }
}

class CustomButton extends StatelessWidget {
  final String title;
  final IconData icon;
  final VoidCallback onClick;

  const CustomButton({
    super.key,
    required this.title,
    required this.icon,
    required this.onClick,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onClick,
      child: Row(
        children: [
          Icon(icon),
          const SizedBox(width: 16),
          Text(title),
        ],
      ),
    );
  }
}