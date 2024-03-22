//
//  Model.swift
//  Ring-project
//
//  Created by Eyoel Gebre on 2/15/24.
//

import Foundation
import onnxruntime_objc
import SwiftUI

class ORTDinoModel {
    private let ortEnv: ORTEnv
    private let ortSession: ORTSession
    let inputShape: [NSNumber] = [1, 3, 224, 224]
    let fileManager = FileManager.default
    let bundleURL = Bundle.main.bundleURL
    
    enum ModelError: Error {
        case Error(_ message: String)
    }
    
    init() throws {
        // Loading the model
        let name = "dinov2_vits14_forward_features_with_transformed"
        ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        guard let modelPath = Bundle.main.path(forResource: name, ofType: "ort") else {
            throw ModelError.Error("Failed to find model file:\(name).ort")
        }
        ortSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: nil)
    }
    
    // Produces the equivalent of a numpy array of RGBA pixel values for a UIImage
    func imageToTensor(img: UIImage) -> [[[Float]]] {
        guard let cgImage = img.cgImage else {
            print("error when getting cgImage")
            return [[[Float(0)]]]
        }
        
        let width = cgImage.width
        let height = cgImage.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        let context = CGContext(data: &pixelData,
                                width: width,
                                height: height,
                                bitsPerComponent: bitsPerComponent,
                                bytesPerRow: bytesPerRow,
                                space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        var rgbData: [[[Float]]] = Array(repeating: Array(repeating: Array(repeating: Float(0), count: width), count: height), count: 3)
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                let pixelIndex = index * bytesPerPixel
                rgbData[0][y][x] = Float(pixelData[pixelIndex])
                rgbData[1][y][x] = Float(pixelData[pixelIndex + 1])
                rgbData[2][y][x] = Float(pixelData[pixelIndex + 2])
            }
        }
        
        return rgbData
    }
    
    // Converts ORTValue to a 3D array of Floats
    func ORTToTensor(ortValue: ORTValue) -> [[[Float]]]? {
        guard let tensorData = try? ortValue.tensorData() as Data else {
            print("Failed to get tensor data from ORTValue")
            return nil
        }
        
        guard let shapeNumbers = try? ortValue.tensorTypeAndShapeInfo().shape else {
          print("ORTValue does not contain a valid shape")
          return nil
        }
        let shape = shapeNumbers.map { Int(truncating: $0) }
        guard shape.count == 3 else {
          print("ORTValue does not contain a 3D tensor")
          return nil
        }
        let totalElements = shape.reduce(1, *)
                guard tensorData.count == totalElements * MemoryLayout<Float>.size else {
            print("Mismatch between expected tensor size and actual data size")
            return nil
        }
        
        let floatValues: [Float] = tensorData.withUnsafeBytes {
            Array(UnsafeBufferPointer(start: $0.baseAddress!.assumingMemoryBound(to: Float.self), count: totalElements))
        }
        
        var threeDArray: [[[Float]]] = Array(repeating: Array(repeating: Array(repeating: 0, count: shape[2]), count: shape[1]), count: shape[0])
        for i in 0..<shape[0] {
            for j in 0..<shape[1] {
                for k in 0..<shape[2] {
                    threeDArray[i][j][k] = floatValues[i * shape[1] * shape[2] + j * shape[2] + k]
                }
            }
        }
        
        return threeDArray
    }
    
    // Computes the 16x16x384 per-patch embedding for a UIImage
    func computeDinoFeat(from_img: UIImage) -> [[[Float]]]? {
        let tensor: [[[Float]]] = imageToTensor(img: from_img)
        let flatTensor = tensor.flatMap { $0.flatMap { $0 } }
        let shape: [NSNumber] = [NSNumber(value: tensor.count), NSNumber(value: tensor[0].count), NSNumber(value: tensor[0][0].count)]
        let data_obj = flatTensor.withUnsafeBufferPointer { Data(buffer: $0) }
        
        do {
            let ortInput = try ORTValue(
                        tensorData: NSMutableData(data: data_obj),
                        elementType: ORTTensorElementDataType.float,
                        shape: shape)

            let output = try ortSession.run(withInputs: ["input_img": ortInput],
                                                     outputNames: ["embeddings"],
                                                     runOptions: nil)
            
            guard let ORTout = output["embeddings"] else {
                print("output was null in Dino run")
                return nil
            }
            
            return ORTToTensor(ortValue: ORTout)
        } catch {
            print("error computing Dino feats: \(error)")
            return nil
        }
    }

    func embMean(emb: [[[Float]]], s: Int, e: Int) -> [Float] {
        let a = emb.count
        let b = emb[0].count
        let c = emb[0][0].count
        
        assert(s >= 0 && s < e && e <= min(a, b), "Invalid range for s and e")
        
        var result = [Float](repeating: 0.0, count: c)
        
        for k in 0..<c {
            var sum: Float = 0.0
            for i in s..<e {
                for j in s..<e {
                    sum += emb[i][j][k]
                }
            }
            result[k] = sum / Float((s - e) * (s - e))
        }
        
        return result
    }
    
    func cosineSimilarity(a: [Float], b: [Float]) -> Float {
        assert(a.count == b.count, "Input arrays must have the same length")
        
        let dotProduct = zip(a, b).map { $0 * $1 }.reduce(0, +)
        let normA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let normB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        
        if normA == 0 || normB == 0 {
            return 0.0
        }
        
        return dotProduct / (normA * normB)
    }
    
    /*
     Nearest Neighbor Algorithm
     
     Args:
         q: embedding with shape [16, 16, 384]
         database: {
             obj1: {
                 'img1': embedding with shape  [16, 16, 384],
                 'img2': embedding with shape  [16, 16, 384],
                 ...
             },
             obj2: {
                 ...
             }
             ...
         }
         patch_len: side length of the square patch
     */
    func computeSim(q: [[[Float]]], database: [String: [String: [[[Float]]]]], patch_len: Int = 4) -> [String : [String : [String : Float]]] {
        // TODO: check whether rounding is OK.
        let start = 8 - (patch_len / 2)
        let end = 8 + Int(ceil(Float(patch_len) / 2.0))
        var sims: [String: [String: [String: Float]]] = [:]
        for (obj, vecs) in database {
            sims[obj] = [:]
            for (img, emb) in vecs {
                // Object similarity using center patchLen x patchLen patch embeddings
                let _q = embMean(emb: q, s: start, e: end)
                let _emb = embMean(emb: emb, s: start, e: end)
                let objSim = cosineSimilarity(a: _q, b: _emb)

                sims[obj]![img] = ["obj": objSim]
            }
        }
        
        return sims
    }
    
    func get_prediction(query_file: String, database: [String: [String: [[[Float]]]]]) -> [(Float, String, String)] {
        
        guard let image = UIImage(named: query_file) else {
            print("image not loaded")
            return []
        }
        guard let q = computeDinoFeat(from_img: image) else {
            print("Could not generate Dino features in get_pred")
            return []
        }
        
        let sims = computeSim(q: q, database: database)
        var scores: [(Float, String, String)] = []
        for (obj, files) in sims {
            for (file, simData) in files {
                if let objsim = simData["obj"] {
                    scores.append((objsim, obj, file))
                }
            }
        }
        
        scores.sort { $0.0 > $1.0 }
        return scores
    }
    
    // Loads the dataset and checks accuracy of the Dino model
    func eval(filepath: String) -> Void {
        // Get the query and ref image file paths
        let queryDirectory = bundleURL.appendingPathComponent("Data/test_set/queries")
        var queryFiles: [String] = []
        let refDirectory = bundleURL.appendingPathComponent("Data/test_set/references")
        var refFiles: [String] = []
        do {
            var subDirectories = try fileManager.contentsOfDirectory(at: queryDirectory, includingPropertiesForKeys: nil)
            for subDirectory in subDirectories {
                let files = try fileManager.contentsOfDirectory(at: subDirectory, includingPropertiesForKeys: nil)
                let jpgFiles = files.filter { $0.pathExtension.lowercased() == "jpg" }
                let filePaths = jpgFiles.map { $0.path }
                queryFiles.append(contentsOf: filePaths)
            }
            queryFiles.sort()
            
            subDirectories = try fileManager.contentsOfDirectory(at: refDirectory, includingPropertiesForKeys: nil)
            for subDirectory in subDirectories {
                let files = try fileManager.contentsOfDirectory(at: subDirectory, includingPropertiesForKeys: nil)
                let jpgFiles = files.filter { $0.pathExtension.lowercased() == "jpg" }
                let filePaths = jpgFiles.map { $0.path }
                refFiles.append(contentsOf: filePaths)
            }
            refFiles.sort()
        } catch {
            print("Error retrieving query or ref files: \(error)")
        }
        
        func getLabel(_ path: String) -> String {
            return String(path.split(separator: "/").suffix(2).first ?? "")
        }
        
        // Create a reference dictionary - Map from label to list of files
        var references: [String: [String]] = [:]
        for file in refFiles {
            let label = getLabel(file)
            if references[label] == nil {
                references[label] = []
            }
            references[label]?.append(file)
        }
        
        // Create a query table
        var queries: [(label: String, pred: String, score: Float)] = []
        
        // Ref embeddings - compute embedding for each image
        var refEmb: [String: [String: [[[Float]]]]] = [:]
        for (label, files) in references {
            refEmb[label] = [:]
            for file in files {
                if let image = UIImage(contentsOfFile: file) {
                    refEmb[label]![file] = computeDinoFeat(from_img: image)
                }
            }
        }
        
        // Predictions
        for queryFile in queryFiles {
            let prediction = get_prediction(query_file: queryFile, database: refEmb)
            if let (score, pred, _) = prediction.first {
                let label = getLabel(queryFile)
                queries.append((label: label, pred: pred, score: score))
            }
        }
        
        for q in queries {
            print("\(q)")
        }
        
        let correct = queries.filter { $0.pred == $0.label }
        let accuracy = 100.0 * Float(correct.count) / Float(queries.count)
        print("Accuracy = \(accuracy)")
    }
}
