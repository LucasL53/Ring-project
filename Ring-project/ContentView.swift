//
//  ContentView.swift
//  Ring-project
//
//  Created by Eyoel Gebre on 2/15/24.
//

import SwiftUI

struct ContentView: View {
    private let model = try! ORTDinoModel()
    
    private func printTensorAndEmbedding() async -> Void {
        if let imagePath = Bundle.main.path(forResource: "IMG_3264", ofType: "JPG", inDirectory: "Data/test_set/queries/blinds_2") {
            guard let img = UIImage(named: imagePath) else {
                print("UIImage() failed")
                return
            }
            let img_ten = model.imageToTensor(img: img)
            print(img_ten)
            print("--------------------------------------------------------------------------------------------")
            guard let tensor = model.computeDinoFeat(from_img: img) else {
                print("couldn't get embs")
                return
            }
            print("tensor dims: \(tensor.count) x \(tensor[0].count) x \(tensor[0][0].count)")
            
            print("[", terminator: "")
            for i in 0..<tensor.count {
                print("[", terminator: "")
                for j in 0..<tensor[i].count {
                    let row = tensor[i][j].map { String($0) }.joined(separator: ", ")
                    print("[\(row)],")
                }
                print("],", terminator: "")
            }
            print("]", terminator: "")
        }
    }

    private func runDino() async -> Void {
        model.eval()
    }

    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
            Button(action: {
                Task {
                    await runDino()
                }
            }) {
                Text("runDino")
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
