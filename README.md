```mermaid
flowchart TD
    %% Main User Flow and System Components
    USER([User]) --> HOME[Home Page]
    HOME --> CONFIG[Configure Simulation]
    CONFIG --> |Set Parameters| WG[Workload Generator]
    WG --> |Generate Tasks| ENV[CPU Environment]
    CONFIG --> |Set Parameters| RL[RL Scheduler]
    CONFIG --> |Set Parameters| TRAD[Traditional Schedulers]

    %% Core System
    subgraph Core["Core System"]
        direction TB
        subgraph Simulation["Simulation Environment"]
            ENV --> |Task Queue| TASK_ASSIGN[Task Assignment]
            TASK_ASSIGN --> CPU_EXEC[CPU Execution]
            CPU_EXEC --> METRICS[Performance Metrics]
            METRICS --> |State| ENV
        end
        
        subgraph Schedulers["CPU Schedulers"]
            RL --> |Select Action| TASK_ASSIGN
            
            subgraph TRAD["Traditional Schedulers"]
                FCFS[First Come First Served]
                SJF[Shortest Job First]
                RR[Round Robin]
            end
            
            TRAD --> |Select Action| TASK_ASSIGN
            FCFS --> TRAD
            SJF --> TRAD
            RR --> TRAD
        end
        
        ENV --> |Current State| RL
        METRICS --> |Reward| RL
    end

    %% Visualization and Results Flow
    METRICS --> COMPARE[Compare Results]
    COMPARE --> VIZ[Visualization]
    VIZ --> |Charts| USER

    %% Database Flow
    COMPARE --> SAVE[Save to Database]
    SAVE --> DB[(Database)]
    DB --> MANAGE[Database Management]
    MANAGE --> |View| USER
    MANAGE --> |Load| CONFIG

    %% API Integration
    subgraph API["API Layer"]
        API_CONFIGS[API Configs]
        API_RESULTS[API Results ID]
        API_METRICS[API Compare Metrics]
    end

    DB --> API_CONFIGS
    DB --> API_RESULTS
    METRICS --> API_METRICS

    API_CONFIGS --> EXT_APP([External Applications])
    API_RESULTS --> EXT_APP
    API_METRICS --> EXT_APP

    %% Training Loop
    CONFIG --> |Start Training| TRAIN[Training Process]
    TRAIN --> |Initialize| RL
    RL --> |Update Q-Values| RL
    TRAIN --> |Episodes Done| COMPARE

    %% Component Descriptions
    classDef blue fill:#2374ab,stroke:#1e5e8a,color:#fff
    classDef green fill:#298e8a,stroke:#1f615e,color:#fff
    classDef orange fill:#ec7505,stroke:#b85a04,color:#fff
    classDef red fill:#a83941,stroke:#7c2b31,color:#fff
    classDef purple fill:#8344ad,stroke:#5f327e,color:#fff

    class HOME,CONFIG,USER,TRAIN blue
    class Core,Simulation,ENV,TASK_ASSIGN,CPU_EXEC,METRICS green
    class Schedulers,RL,TRAD,FCFS,SJF,RR orange
    class COMPARE,VIZ,SAVE,DB,MANAGE red
    class API,API_CONFIGS,API_RESULTS,API_METRICS,EXT_APP,WG purple

    %% Legend
    subgraph Legend
        L1[User Interface] --- L2[Simulation System] --- L3[Schedulers] --- L4[Database] --- L5[API Integration]
    end

    class L1 blue
    class L2 green
    class L3 orange
    class L4 red
    class L5 purple

