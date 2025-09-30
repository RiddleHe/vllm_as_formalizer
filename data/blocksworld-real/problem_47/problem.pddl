(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        orange - block
        purple - block
        yellow - block
        red - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (clear green)
        (ontable orange)
        (clear orange)
        (ontable purple)
        (clear purple)
        (ontable yellow)
        (clear yellow)
        (ontable red)
        (on blue red)
        (clear blue)
        ; green
        ; orange
        ; purple
        ; yellow
        ; red -> blue
    )
    (:goal (and (on yellow orange)
        (on blue yellow)
        (on red blue)
        (on purple red)
        (on green purple)))
)
    