(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        purple - block
        blue - block
        orange - block
        red - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (clear green)
        (ontable purple)
        (clear purple)
        (ontable blue)
        (on orange blue)
        (clear orange)
        (ontable red)
        (on yellow red)
        (clear yellow)
        ; green
        ; purple
        ; blue -> orange
        ; red -> yellow
    )
    (:goal (and (on green blue)
        (on orange red)
        (on yellow purple)))
)
    