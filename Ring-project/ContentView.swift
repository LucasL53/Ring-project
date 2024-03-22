//
//  ContentView.swift
//  Ring-project
//
//  Created by Eyoel Gebre on 2/15/24.
//

import SwiftUI

struct ContentView: View {
    private let model = try! ORTDinoModel()
    
    private func runDino(file: String) async -> Void {
        model.eval(filepath: file)
    }

    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
            Button(action: {
                Task {
                    await runDino(file: "filename")
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
