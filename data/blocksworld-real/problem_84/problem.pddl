(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        blue - block
        yellow - block
        green - block
        red - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (clear orange)
        (ontable blue)
        (clear blue)
        (ontable yellow)
        (clear yellow)
        (ontable green)
        (clear green)
        (ontable red)
        (on purple red)
        (clear purple)
        ; orange
        ; blue
        ; yellow
        ; green
        ; red -> purple
    )
    (:goal (and (on orange purple)
        (on yellow orange)
        (on red yellow)
        (on blue red)
        (on green blue)))
)
    